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
| real robot | `robot_io` (native, owns camera + motors) | ZMQ :5555, the protocol of gym-duckietown's `Duckiebot-v0` (which expects :7777: use `--port 7777` for it) |
| real robot | agent in the Ultralytics JetPack container (TensorRT FP16) or on a laptop | `RealRobotEnv.step([v, ω])` |

## Results (simulation)

**Model.** YOLO11n-seg at 320 px, trained for 8 epochs on 21,951 auto-labelled frames from 16 maps,
with domain and camera randomization on. Mask mAP50:

| split | all | yellow | white | red stop | duckie | duckiebot |
|---|---|---|---|---|---|---|
| validation (training maps, held-out episodes) | 0.634 | 0.622 | 0.709 | 0.695 | 0.620 | 0.524 |
| test (2 maps never used for training) | 0.657 | 0.654 | 0.665 | 0.653 | – | – |

Inference on an Apple M4 Max: 11 ms/frame on the CPU, 29 ms on MPS (including pre/post-processing).

**Closed loop** in gym-duckietown, 800 steps (≈27 s) per episode, 3 episodes per map, same seeds
for every agent (`duckie_yolo.sim.eval_sim`). *expert* = privileged pure pursuit on the
simulator's true pose. *oracle* = this agent fed perfect masks from the segmentation render.
*color* = HSV thresholds instead of YOLO.

Lane following, domain randomization on (`loop_empty`, `small_loop`, `zigzag_dists`, plus the unseen
`ETHZ_autolab_technical_track`):

| agent | survived | distance / episode | mean \|d\| | mean \|φ\| |
|---|---|---|---|---|
| expert (privileged) | 12/12 | 4.4 m | 2.3 cm | 6.8° |
| oracle (perfect masks) | 10/12 | 6.0 m | 7.0 cm | 10.7° |
| **YOLO11n-seg** | **10/12** | **6.0 m** | **7.1 cm** | **10.8°** |
| color thresholds | 10/12 | 1.3 m (mostly stopped: lane lost) | 15.1 cm | 17.0° |

Intersections and obstacles (`4way`, `udem1`, unseen `ETH_large_intersect`, `loop_obstacles`,
`loop_dyn_duckiebots`), starting in lane (`--start-in-lane`):

| agent | survived | red-line stops | obstacle stops | notes |
|---|---|---|---|---|
| expert (privileged) | 11/15 | 0 | 0 | no stop-line / obstacle logic: drives into obstacles |
| oracle (perfect masks) | 13/15 | 14 | 4 | 2 failures in `4way` |
| **YOLO11n-seg** | **15/15** | **17** | **4** | stops at red lines, crosses, stops for duckies / duckiebots |

So the YOLO perception is as good as perfect masks for driving. The remaining gap to the
privileged expert is the geometric estimator, not the model. Limitations seen in simulation:
* It **waits behind static obstacles** indefinitely, because there is no overtaking.
* Random starts outside the lane (in the opposite lane, or turned more than about 30°) are often not recovered.

Three details matter here, all handled in the code:
* **Simulator dynamics.** gym-duckietown (daffy) integrates identified motor dynamics. Realized
  speed is 0.70× and realized yaw rate 0.45× the commanded values, so a geometric controller
  under-steers by about 36% unless this is compensated (`ActuationConfig`,
  `tools/identify_sim_dynamics.py`). Without compensation the oracle survived 2/9 episodes;
  with it, 8/9. On the real robot the same gains come from `calibrate_wheels.py`.
* **Intersections.** After stopping, the robot crosses along a virtual lane whose direction comes
  from the red stop line (perpendicular to the lane), dead-reckoned through the intersection.
  A single visible white line is disambiguated (our right edge vs. the road's left edge)
  using the lane being tracked.
* **Segmentation alignment.** The upstream segmentation render used different tile textures than
  the RGB render, which misplaced dash labels. This is fixed in this fork (see below).

Watch it: `python -m duckie_yolo.sim.run_sim --map udem1 --weights models/best.pt --device cpu --start-angle 15 --top-down`.
`--top-down` adds a bird's-eye panel with the robot, the lane-centre path it follows (green) and its steering target (red).

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

### Linux / Windows (WSL2)

On Windows, use **WSL2 with Ubuntu 22.04**. On Windows 11, WSLg shows the simulator windows
directly. Native Windows is not supported. In PowerShell: `wsl --install -d Ubuntu-22.04`, then `wsl --update`.

```bash
sudo apt update && sudo apt install -y git curl build-essential libgl1 libgl1-mesa-dri libglu1-mesa \
    libegl1 mesa-utils libglib2.0-0 libsm6 libxext6 libxrender1 xvfb
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

git clone -b yolo-lane-following https://github.com/SanchakGarg/gym-duckietown.git
cd gym-duckietown
uv venv --python 3.8 .venv && source .venv/bin/activate
uv pip install "pip<24.1" "setuptools<66" "wheel<0.40"
# no NVIDIA GPU in WSL? install the small CPU build of PyTorch first:
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r yolo_lane_following/requirements-sim.txt
python -m pip install --no-build-isolation --no-deps -e . -e yolo_lane_following
pytest yolo_lane_following/tests

# trained weights are not in git: put best.pt into yolo_lane_following/models/
cd yolo_lane_following
export DUCKIETOWN_PYGLET_HEADLESS=0      # render through the X display (WSLg) instead of EGL
python -m duckie_yolo.sim.run_sim --map udem1 --weights models/best.pt --device cpu --start-angle 15 --top-down
```
* **No window** (no WSLg, or a server): run `xvfb-run -a -s "-screen 0 1280x1024x24" python -m duckie_yolo.sim.run_sim ... --no-show --video out.mp4`.
* **NVIDIA GPU:** use `--device 0` when `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
* **Talking to the robot from WSL:** use its IP address, because `<name>.local` usually does not resolve inside WSL.
* **ROS sourced in your shell** (e.g. `/opt/ros/jazzy/setup.bash` in `~/.bashrc`): its Python 3.12 packages land on
  `PYTHONPATH` and break the Python 3.8 venv. pytest, for example, fails with `No module named 'lark'` from `launch_testing`.
  Run `echo 'unset PYTHONPATH' >> .venv/bin/activate` once, or use a shell without ROS. duckie_yolo itself never uses ROS.

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

# (c) train YOLO11n-seg, exports ONNX. On Apple Silicon --device cpu is ~2.5x faster than mps
#     for segmentation training: 8 epochs over the 21,951 frames took ~1.5 h on an M4 Max
python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --epochs 8 --close-mosaic 2 --device cpu --name sim_v1
mkdir -p models && cp runs/segment/sim_v1/weights/best.pt models/

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

## 3. Real Duckiebot (Jetson Nano / Orin Nano, no ROS)

**Hardware path.** Taken from Duckietown's `dt-duckiebot-interface` as reference only;
nothing from it is imported or run.
* Wheels: HATv3 → PCA9685 at I2C **0x60 on bus 7** (Orin Nano; **bus 1 on Jetson Nano**: pass `--i2c-bus 1`), 1600 Hz.
  * Left motor: direction on PCA9685 channels 10/9, speed on channel 8.
  * Right motor: direction on Jetson GPIO **BOARD 33/31**, speed on channel 13.
  * Duty cycles map to `pwm_min 60 … pwm_max 255`, with deadzone 0.01.
  * Drivers: `duckie_yolo/robot/hw/pca9685.py` (`smbus2`) and `hw/motors.py` (`Jetson.GPIO`).
* Camera: IMX219 through `nvarguscamerasrc` (1640×1232@30 → 640×480), using OpenCV-GStreamer or a
  raw `gst-launch … ! fdsink` pipe (`hw/camera.py`).

```bash
# on the robot (clone your fork there)
git clone -b yolo-lane-following https://github.com/SanchakGarg/gym-duckietown.git
cd gym-duckietown/yolo_lane_following
bash duckie_yolo/robot/setup_orin.sh    # asks before every change: Python env, robot_io service, swap/clocks, web UI container

# LIFT THE ROBOT: spins each wheel, saves a camera frame (stop the robot_io service first if it runs)
~/dy/bin/python -m duckie_yolo.robot.robot_io --i2c-bus 1 --hw-test
```
`setup_orin.sh` installs `robot_io` (camera + wheels on ZMQ :5555, watchdog 0.5 s, duty cap 0.8) as the
`duckie-robot-io` service in a Python ≥ 3.8 env (`~/dy`). On **JetPack 4 (Jetson Nano, Ubuntu 18.04,
Python 3.6)** it creates that env with `python3.8` and pins `Jetson.GPIO<2.1` (2.1+ needs the gpiochip
interface of newer kernels) and `opencv-python-headless<5` (prebuilt wheels only; without GStreamer
the camera uses a `gst-launch` pipe).

### Web interface (browser only)
Everything below can be done from a browser at **`http://<robot-ip>:8080`**; nothing runs on the laptop.
The web UI runs on the robot in the GPU Docker container (Ultralytics + TensorRT for the JetPack) and is
`robot_io`'s only client:
```bash
bash duckie_yolo/robot/webui/start_webui.sh   # build image if needed, (re)start container; restarts at boot
docker logs -f duckie-webui                   # its log
```
| tab | what |
|---|---|
| **Drive** | live camera (raw / corrected / model view), hold-to-drive with W A S D or on-screen buttons, **record a dataset while driving** (R) |
| **Model** | pick a model in `models/` (or colour thresholds), run it on the Jetson, "let the model drive" with a speed cap, upload models, build a TensorRT engine |
| **Datasets** | recorded sessions (`data/real/<time>_<name>/raw` + `rect`): download as zip, re-correct after calibrating, delete |
| **Calibrate** | lens (auto-captures checkerboard views), floor (x0 / y0, grid check image), wheels (straight + spin tests), printable checkerboard |
| **Guidelines** | what to record, red stop lines, label classes and rules, the fine-tuning loop |

Safety: the robot only moves while the page sends commands. Teleop stops 0.3 s after the last key
event, autodrive needs a heartbeat from the open page (1 s), STOP / Space / Esc cancels everything,
and `robot_io`'s watchdog and duty cap stay in force. The container has no memory or CPU limits (the
Jetson GPU shares the board's RAM); `setup_orin.sh` can add swap, max clocks and a headless boot to
leave the model as much memory as possible. Without Docker, `python -m duckie_yolo.robot.webui` runs
the same page with colour thresholds only.

### Command-line tools (alternative)
Watch the camera from a laptop (the wheels never move; `robot_io` serves one client at a time):
`python -m duckie_yolo.robot.view_camera --host <robot>.local [--robot-config configs/robot_mybot.yaml]`.

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

**Drive.** Run this on the Jetson; inference uses TensorRT FP16.
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
# keys (w/s/a/d, space, r = record, q) are read in the terminal and in the camera window;
# the terminal works where the window gets no key events (e.g. WSLg). --no-window: terminal only
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
  robot/                robot_io.py, real_env.py, run_robot.py, view_camera.py, calibrate_*.py, teleop_collect.py,
                        autolabel_real.py, export_engine.py, setup_orin.sh, docker/Dockerfile.agent, hw/,
                        webui/ (browser interface: controller, recorder, models, calibration, static page)
  train.py              Ultralytics training + export
configs/                agent.yaml, agent_real.yaml, robot_default.yaml
tests/                  geometry/estimator, labels, control/FSM, PCA9685/motors, robot_io <-> client over TCP,
                        web UI (teleop/deadman, autodrive heartbeat, recording, uploads, calibration functions)
```

## Status / limitations
* The hardware path (I2C/GPIO pins, camera pipeline) mirrors Duckietown's DB21/DB26J configuration
  and is unit-tested against mocks. `robot_io` and the camera have been brought up on a DB21
  (Jetson Nano, JetPack 4.6); the web UI's Docker image on JetPack 4 and TensorRT engines on the
  Nano are not verified yet. Lift the robot for the first `--hw-test`, and test models in
  *Model view* before letting them drive.
* Intersections are handled only as "stop at the red line, then cross straight". There is no
  turn planning.
* The default extrinsic board offsets (`--x0 0.191 --y0 -0.093`) are a starting point. Measure
  your own setup.
