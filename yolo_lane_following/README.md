# duckie_yolo: YOLO lane following for Duckietown (sim → real, no ROS)

A Duckiebot follows the road using a **YOLO11n-seg** model that segments lane markings. The
model is trained entirely on **auto-labelled gym-duckietown images**. The **same agent code**
drives the simulator and a physical Duckiebot (tested target: **4 GB Jetson Orin Nano / DB26J**)
**without ROS or the Duckietown software stack**.

```
camera ─► (rectify, real robot only) ─► YOLO11n-seg ─► masks: yellow / white / red / duckie / duckiebot
       ─► ground projection (homography) ─► RANSAC fits of yellow & white lines ─► lane centre y_c(x)
       ─► pure pursuit (lookahead 0.3 m) ─► behaviour FSM (DRIVE / STOP_LINE_WAIT / CROSS / OBSTACLE_STOP / LOST)
       ─► (v, ω) ─► DuckietownEnv kinematics ─► wheel duty cycles
```

| where | what runs | how it talks to the robot |
|---|---|---|
| simulator | `gym_duckietown.envs.DuckietownEnv` | `env.step([v, ω])` |
| real robot | `robot_io` (native, owns camera + motors) | ZMQ :7777, the protocol of gym-duckietown's `Duckiebot-v0` |
| real robot | agent in the Ultralytics JetPack container (TensorRT FP16) or on a laptop | `RealRobotEnv.step([v, ω])` |

## Results (simulation)

Closed loop in gym-duckietown, 800 steps (≈27 s) per episode, random starts:

| agent | loop_empty | small_loop | zigzag_dists | mean lateral error |
|---|---|---|---|---|
| perfect masks ("oracle": tests geometry + control) | 2/3 | 3/3 | 3/3 | 3–5 cm |
| YOLO11n-seg (sim-trained) | _pending_ | | | |

Two details matter here, both handled in the code:
* **Simulator dynamics.** gym-duckietown (daffy) integrates identified motor dynamics. Realized
  speed is 0.70× and realized yaw rate 0.45× the commanded values, so a geometric controller
  under-steers by about 36% unless this is compensated (`ActuationConfig`,
  `tools/identify_sim_dynamics.py`). Without compensation the oracle survived 2/9 episodes;
  with it, 8/9. On the real robot the same gains come from `calibrate_wheels.py`.
* **Segmentation alignment.** The upstream segmentation render used different tile textures than
  the RGB render, which misplaced dash labels. This is fixed in this fork (see below).

## 1. Install (development machine)

Tested on macOS (Apple Silicon, Python 3.8). Linux works the same way. For headless Linux use
Xvfb and `DUCKIETOWN_PYGLET_HEADLESS=0`.

```bash
cd gym-duckietown
uv venv --python 3.8 .venv && source .venv/bin/activate     # or python3.8 -m venv
uv pip install "pip<24.1" "setuptools<66" "wheel<0.40"
python -m pip install -r yolo_lane_following/requirements-sim.txt
python -m pip install --no-build-isolation --no-deps -e . -e yolo_lane_following
pytest yolo_lane_following/tests
```

Notes:
* Working over SSH on macOS: the simulator has no window server there (pyglet finds no
  screens). Run it from a terminal on the Mac itself. Alternatively, launch the command inside
  the logged-in GUI session (e.g. via a small `osacompile` applet and `open`), keep the display
  awake with `caffeinate -d -u`, and keep that copy outside `~/Desktop` / `~/Documents`, which
  macOS privacy protection blocks for such processes.
* daffy's gym-duckietown needs `gym>=0.22` (it calls `np_random.integers`). `numpy<1.24`
  and `pyglet 1.5.x` are required.
* On macOS the simulator needs a logged-in GUI session with the display awake. It cannot
  render from a plain SSH session.

### Simulator fixes in this fork (`src/gym_duckietown`)
* `check_hw.py` forced pyglet into EGL headless mode on every OS, so the simulator never started on macOS.
* Segment renders of maps with objects crashed: unhashable colour lists were passed to the `lru_cache`d `load_texture`.
* `get_texture` failed with the `numpy.random.Generator` that gym ≥ 0.22 provides.
* **Segmentation textures did not match the RGB render.** The segmented tile texture was resolved by tile *kind*, which picked a different image file, so the segmented yellow dashes were shifted by up to a dash. It is now derived from the exact file used for the RGB render, which makes the segmentation pass usable as ground truth.

## 2. Simulation: data → training → evaluation

```bash
cd yolo_lane_following

# (a) auto-labelled dataset: expert driving + noise, domain & camera randomization, 16 maps
python -m duckie_yolo.sim.generate_dataset --out data/sim_v1 --frames 20000 --test-frames 1000
python tools/check_label_alignment.py data/align      # optional sanity check (--no-domain-rand set)

# (b) closed-loop check of geometry + control with perfect masks (no YOLO yet)
python -m duckie_yolo.sim.eval_sim --agent oracle --maps loop_empty small_loop zigzag_dists --episodes 3

# (c) train YOLO11n-seg (MPS / CUDA / CPU picked automatically), exports ONNX
python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --epochs 60 --name sim_v1

# (d) drive with YOLO: live window, or a video
python -m duckie_yolo.sim.run_sim --map small_loop --weights runs/segment/sim_v1/weights/best.pt
python -m duckie_yolo.sim.run_sim --map loop_obstacles --weights ... --video loop_obstacles.mp4 --no-show

# (e) closed-loop evaluation (also on held-out maps), compare with the privileged expert
python -m duckie_yolo.sim.eval_sim --agent yolo --weights runs/segment/sim_v1/weights/best.pt \
    --maps loop_empty small_loop zigzag_dists ETHZ_autolab_technical_track --episodes 5 --domain-rand
python -m duckie_yolo.sim.eval_sim --agent expert --maps ...
```

How labels are produced: `Simulator.render_obs(segment=True)` renders the same pose with the
asphalt blacked out and the objects flat-coloured. Lane classes come from hue/saturation
thresholds, objects from the simulator's per-mesh colour hash. `duckie` and `duckiebot` share
a colour, so duckiebots are separated by re-rendering with them hidden. Masks become YOLO-seg
polygons (`duckie_yolo/sim/labels.py`).

Flags: `--no-stoplines`, `--no-obstacles` (plain lane following), `--speed`, `--agent-config configs/agent.yaml`.

## 3. Real Duckiebot (Jetson Orin Nano, no ROS)

**Hardware path.** Taken from Duckietown's `dt-duckiebot-interface` as reference only;
nothing from it is imported or run.
* Wheels: HATv3 → PCA9685 at I2C **0x60 on bus 7** (Orin Nano; bus 1 on Jetson Nano), 1600 Hz.
  * Left motor: direction on PCA9685 channels 10/9, speed on channel 8.
  * Right motor: direction on Jetson GPIO **BOARD 33/31**, speed on channel 13.
  * Duty cycles map to `pwm_min 60 … pwm_max 255`, with deadzone 0.01.
  * Drivers: `duckie_yolo/robot/hw/pca9685.py` (`smbus2`) and `hw/motors.py` (`Jetson.GPIO`).
* Camera: IMX219 through `nvarguscamerasrc` (1640×1232@30 → 640×480), using OpenCV-GStreamer or a
  raw `gst-launch … ! fdsink` pipe (`hw/camera.py`).

```bash
# on the robot (clone your fork there)
cd gym-duckietown/yolo_lane_following
bash duckie_yolo/robot/setup_orin.sh          # asks before every change; lists Duckietown containers holding camera/I2C

python3 -m duckie_yolo.robot.robot_io --hw-test        # LIFT THE ROBOT: spins each wheel, saves a camera frame
python3 -m duckie_yolo.robot.robot_io                  # serve camera + wheels on :7777 (watchdog 0.5 s, duty cap 0.8)
```

**Calibrate.** This can run from a laptop with `--host <robot>.local`.
```bash
cp configs/robot_default.yaml configs/robot_mybot.yaml
python -m duckie_yolo.robot.calibrate_intrinsics --host <robot> --capture calib/intr --robot-config configs/robot_mybot.yaml
python -m duckie_yolo.robot.calibrate_extrinsics --host <robot> --robot-config configs/robot_mybot.yaml --x0 <m> --y0 <m>
python -m duckie_yolo.robot.calibrate_wheels    --host <robot> --robot-config configs/robot_mybot.yaml   # then --drift/--distance --save
```
* Intrinsics use a fisheye model. The raw image is rectified to a virtual pinhole camera with the
  simulator's intrinsics (640×480, 75° vertical FOV), so YOLO sees sim-like geometry.
* Extrinsics: lay the checkerboard flat in front of the robot. `--x0` is the distance from the
  axle to the nearest corner row, `--y0` the position of the right-most corner column. The
  script writes `*_extrinsics_check.jpg` with a 10 cm floor grid for checking.

**Drive.** Run this on the Orin; inference uses TensorRT FP16.
```bash
mkdir -p models && cp <best.pt from training> models/best.pt
docker run --rm -it --runtime nvidia --ipc host -v $PWD:/work -w /work --entrypoint python3 \
    duckie-yolo-agent -m duckie_yolo.robot.export_engine --weights models/best.pt     # -> models/best.engine
docker run --rm -it --runtime nvidia --network host --ipc host -v $PWD:/work -w /work duckie-yolo-agent \
    --weights models/best.engine --robot-config configs/robot_mybot.yaml --agent-config configs/agent_real.yaml --dry-run
# watch http://<robot>:8080 ; when the overlay looks right, drop --dry-run (start with --max-speed 0.2)
```
To run the agent on a laptop instead, use
`python -m duckie_yolo.robot.run_robot --host <robot>.local --encoding jpeg --weights best.pt ...`.
The repo's own `Duckiebot-v0` gym env and `manual_control.py` can also connect to `robot_io`.

**Sim → real fine-tuning** (recommended):
```bash
python -m duckie_yolo.robot.teleop_collect --host <robot>.local --robot-config configs/robot_mybot.yaml --out data/real/raw
python -m duckie_yolo.robot.autolabel_real --images data/real/raw --weights best.pt --out data/real/labeled
# fix the masks in CVAT / Label Studio (YOLO segmentation format), then:
python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --extra-train data/real/labeled/images \
    --model runs/segment/sim_v1/weights/best.pt --epochs 30 --name sim_real_v1
```

## Layout
```
duckie_yolo/
  agent.py              LaneFollowingAgent: image -> (v, ω)        config.py   typed config (YAML)
  perception/           camera_model.py (rectify, homography), segmenters.py (YOLO / HSV / label map)
  estimation/           lane_estimator.py (ground points, RANSAC fits, lane centre, stop line, obstacles)
  control/              pure_pursuit.py, behavior.py (FSM), kinematics.py (= DuckietownEnv.step)
  sim/                  generate_dataset.py, labels.py, expert.py, run_sim.py, eval_sim.py
  robot/                robot_io.py, real_env.py, run_robot.py, calibrate_*.py, teleop_collect.py,
                        autolabel_real.py, export_engine.py, setup_orin.sh, docker/Dockerfile.agent, hw/
  train.py              Ultralytics training + export
configs/                agent.yaml, agent_real.yaml, robot_default.yaml
tests/                  geometry/estimator, labels, control/FSM, PCA9685/motors, robot_io <-> client over TCP
```

## Status / limitations
* The hardware path (I2C/GPIO pins, camera pipeline) mirrors Duckietown's DB26J configuration
  and is unit-tested against mocks, but **it has not been run on a physical robot by the author
  of this code**. Lift the robot for the first `--hw-test`, and start in `--dry-run`.
* Intersections are handled only as "stop at the red line, then cross straight". There is no
  turn planning.
* The default extrinsic board offsets (`--x0 0.191 --y0 -0.093`) are a starting point. Measure
  your own setup.
