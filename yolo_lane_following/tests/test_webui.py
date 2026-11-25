"""Web UI against an in-process robot_io (mock camera + fake motors) over real TCP and HTTP."""
import argparse
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np
import pytest

zmq = pytest.importorskip("zmq")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait(cond, timeout=5.0, dt=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(dt)
    return False


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    from duckie_yolo.robot import robot_io
    from duckie_yolo.robot.hw.camera import MockCamera
    from duckie_yolo.robot.hw.motors import fake_wheels
    from duckie_yolo.robot.webui.calibration import Calibration
    from duckie_yolo.robot.webui.controller import RobotController
    from duckie_yolo.robot.webui.models import JobRunner, ModelManager
    from duckie_yolo.robot.webui.recorder import Recorder
    from duckie_yolo.robot.webui.server import WebUI, serve

    rport, wport = _free_port(), _free_port()
    monkeypatch.setattr(robot_io.signal, "signal", lambda *a, **k: None)
    wheels, _, _ = fake_wheels("v3")
    cam = MockCamera(640, 480, 30).start()
    args = argparse.Namespace(bind="127.0.0.1", port=rport, watchdog=0.3, frame_timeout=0.2, max_duty=0.8)
    threading.Thread(target=lambda: robot_io.serve(args, wheels, cam), daemon=True).start()

    models_dir = tmp_path / "models"
    models = ModelManager(str(models_dir), "")
    rec = Recorder(str(tmp_path / "data"))
    ctl = RobotController("127.0.0.1", rport, str(tmp_path / "robot.yaml"), models, rec, "raw", 15.0).start()
    app = WebUI(ctl, Calibration(ctl, str(tmp_path / "calib")), JobRunner(), workdir=str(tmp_path))
    httpd = serve(app, "127.0.0.1", wport)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    assert _wait(lambda: ctl.connected, 5.0), ctl.robot_error

    class Stack:
        pass

    st = Stack()
    st.url, st.ctl, st.wheels, st.app, st.tmp = f"http://127.0.0.1:{wport}", ctl, wheels, app, tmp_path
    yield st
    httpd.shutdown()
    ctl.shutdown()
    cam.stop()


def call(st, method, path, body=None, raw=False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(st.url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out = r.read()
            return (r.status, out) if raw else (r.status, json.loads(out or b"{}"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_status_page_and_static(stack):
    code, s = call(stack, "GET", "/api/status")
    assert code == 200 and s["robot"]["connected"] and s["view"] == "rect"
    code, html = call(stack, "GET", "/", raw=True)
    assert code == 200 and b"Duckiebot" in html and b"/static/app.js" in html
    assert call(stack, "GET", "/static/app.js", raw=True)[0] == 200
    assert call(stack, "GET", "/static/../server.py", raw=True)[0] == 404
    code, svg = call(stack, "GET", "/checkerboard.svg", raw=True)
    assert code == 200 and svg.count(b"<rect") == 1 + 24  # background + half of 8x6 squares


def test_teleop_reaches_wheels_and_deadman_stops(stack):
    call(stack, "POST", "/api/drive", {"v": 0.3, "w": 0.0})
    assert _wait(lambda: stack.wheels.last[0] > 0 and stack.wheels.last[1] > 0, 1.0)
    assert _wait(lambda: stack.ctl.source == "teleop", 0.2)
    # no more commands: the web UI's 0.3 s deadman zeroes the command (robot_io keeps getting zeros)
    assert _wait(lambda: stack.wheels.last == (0.0, 0.0), 1.5)
    assert _wait(lambda: stack.ctl.source == "idle", 0.2)


def test_stop_cancels_teleop_and_motion(stack):
    call(stack, "POST", "/api/calib/wheels/run", {"kind": "straight", "v": 0.25, "duration": 5})
    assert stack.ctl.status()["motion"] is not None
    call(stack, "POST", "/api/stop")
    assert stack.ctl.status()["motion"] is None
    time.sleep(0.2)
    assert stack.wheels.last == (0.0, 0.0)


def test_calibration_motion_runs_after_delay(stack):
    stack.app.calib.wheels_run("spin", omega=2.0, duration=0.4, delay=0.2)
    assert _wait(lambda: stack.ctl.source == "calib", 1.0)
    l, r = stack.wheels.last
    assert r > 0 > l  # spin left in place
    assert _wait(lambda: stack.ctl.source == "idle", 1.5)


def test_autodrive_needs_model_and_heartbeat(stack):
    code, err = call(stack, "POST", "/api/autodrive", {"enabled": True})
    assert code == 400 and "model" in err["error"]
    assert call(stack, "POST", "/api/model/start", {"name": "__color__", "conf": 0.3})[0] == 200
    assert _wait(lambda: stack.ctl.models.state == "running", 10.0), stack.ctl.models.error
    assert call(stack, "POST", "/api/autodrive", {"enabled": True, "max_speed": 0.15})[0] == 200
    for _ in range(5):  # page alive
        call(stack, "POST", "/api/heartbeat")
        time.sleep(0.2)
    assert stack.ctl.autodrive and stack.ctl.source == "auto"
    assert abs(stack.ctl.cmd[0]) <= 0.15 + 1e-9
    # page gone: heartbeats stop -> autodrive switches itself off
    assert _wait(lambda: not stack.ctl.autodrive, 2.5)
    assert "lost contact" in (stack.ctl.event or "")
    # STOP also switches it off
    call(stack, "POST", "/api/autodrive", {"enabled": True})
    call(stack, "POST", "/api/stop")
    assert not stack.ctl.autodrive
    assert call(stack, "POST", "/api/view", {"view": "model"})[0] == 200
    stack.ctl.viewer_enter()
    try:
        assert _wait(lambda: stack.ctl.frames.latest() is not None, 2.0)
        jpg = stack.ctl.frames.latest()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] > 640  # model view = camera + bird's-eye panel
    finally:
        stack.ctl.viewer_exit()
    call(stack, "POST", "/api/model/stop")
    assert stack.ctl.models.state == "idle"


def test_record_list_zip_recorrect_delete(stack):
    code, r = call(stack, "POST", "/api/record/start", {"name": "loop day!", "hz": 10})
    assert code == 200 and r["session"].endswith("_loop-day")
    sess = r["session"]
    call(stack, "POST", "/api/drive", {"v": 0.2, "w": 0.0})  # recording during teleop
    time.sleep(0.6)
    call(stack, "POST", "/api/record/stop")
    code, lst = call(stack, "GET", "/api/sessions")
    s = [x for x in lst["sessions"] if x["name"] == sess][0]
    assert s["raw"] >= 3 and s["raw"] == s["rect"] and s["meta"]["frames"] == s["raw"]
    assert s["meta"]["calibration"]["calibrated"]["intrinsics"] is False

    code, data = call(stack, "GET", f"/api/sessions/{sess}.zip", raw=True)
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    assert f"{sess}/meta.json" in names
    assert sum(n.startswith(f"{sess}/raw/") for n in names) == s["raw"]
    img = cv2.imdecode(np.frombuffer(z.read([n for n in names if "/rect/" in n][0]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (480, 640, 3)

    code, r = call(stack, "POST", f"/api/sessions/{sess}/recorrect")
    assert code == 200 and r["frames"] == s["raw"]
    assert call(stack, "DELETE", "/api/sessions/../../etc")[0] in (400, 404)
    assert call(stack, "DELETE", f"/api/sessions/{sess}")[0] == 200
    assert not (stack.tmp / "data" / sess).exists()


def test_model_upload_list_delete(stack):
    payload = os.urandom(3000)
    req = urllib.request.Request(stack.url + "/api/models/upload?name=my_model.pt", data=payload, method="PUT")
    with urllib.request.urlopen(req) as r:
        assert json.loads(r.read())["name"] == "my_model.pt"
    assert (stack.tmp / "models" / "my_model.pt").read_bytes() == payload
    names = [m["name"] for m in call(stack, "GET", "/api/models")[1]["models"]]
    assert names[0] == "__color__" and "my_model.pt" in names
    req = urllib.request.Request(stack.url + "/api/models/upload?name=../evil.pt", data=b"x", method="PUT")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(req)
    assert call(stack, "DELETE", "/api/models/my_model.pt")[0] == 200
    assert not (stack.tmp / "models" / "my_model.pt").exists()


def test_job_runner_logs_and_exit_code(tmp_path):
    from duckie_yolo.robot.webui.models import JobRunner

    done = []
    j = JobRunner()
    j.start("demo", [sys.executable, "-c", "import sys; print('hello'); sys.stdout.write('50%\\r100%\\n'); sys.exit(3)"],
            on_done=done.append)
    assert _wait(lambda: j.state != "running", 10.0)
    st = j.status(with_log=True)
    assert st["state"] == "failed" and st["returncode"] == 3 and done == [3]
    assert "hello" in st["log"] and "100%" in st["log"] and "50%" not in st["log"]
    with pytest.raises(RuntimeError):
        j.start("a", [sys.executable, "-c", "import time; time.sleep(5)"])
        j.start("b", [sys.executable, "-c", "pass"])
    j.cancel()
    assert _wait(lambda: j.state == "cancelled", 5.0)


# ---------------------------------------------------------------------- calibration functions
def _render_board(K, D, rvec, tvec, cols=7, rows=5, square=0.031, size=(640, 480)):
    """Synthetic camera image of a checkerboard (white margin) seen from pose rvec/tvec."""
    nx, ny = cols + 1, rows + 1
    px = 40
    board = np.full(((ny + 2) * px, (nx + 2) * px), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                board[(j + 1) * px:(j + 2) * px, (i + 1) * px:(i + 2) * px] = 0
    # board pixel (u, v) -> metric point (x, y, 0); inner corner (0,0) sits at board pixel (2px, 2px).
    # A planar board under a pinhole camera is a homography: project its 4 outer corners and warp.
    s = square / px
    corners_b = np.array([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]], np.float64)
    obj = np.c_[(corners_b - 2 * px) * s, np.zeros(4)]
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    H = cv2.getPerspectiveTransform(corners_b.astype(np.float32), img_pts.reshape(-1, 2).astype(np.float32))
    warped = cv2.warpPerspective(board, H, size, borderValue=128)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def test_calibrate_images_recovers_pinhole_camera(tmp_path):
    from duckie_yolo.robot.calibrate_intrinsics import calibrate_images

    K = np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]])
    rng = np.random.default_rng(0)
    files = []
    for i in range(14):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.12, 0.02), rng.uniform(-0.1, 0.0), rng.uniform(0.35, 0.6)])
        p = str(tmp_path / f"v{i:02d}.png")
        cv2.imwrite(p, _render_board(K, np.zeros(5), rvec, tvec))
        files.append(p)
    cfg = str(tmp_path / "robot.yaml")
    r = calibrate_images(files, cfg, model="pinhole")
    assert r["views"] >= 8 and r["rms"] < 1.0
    assert abs(r["K"][0, 0] - 420) < 15 and abs(r["K"][0, 2] - 320) < 15
    from duckie_yolo.robot.robot_config import RobotConfig

    rc = RobotConfig.load(cfg)
    assert rc.calibrated["intrinsics"] and not rc.calibrated["extrinsics"]
    assert os.path.exists(r["sample"])


def test_compute_extrinsics_recovers_ground_points():
    from duckie_yolo.robot.calibrate_extrinsics import compute_extrinsics, ground_grid
    from duckie_yolo.robot.robot_config import RobotConfig

    robot = RobotConfig()  # default (sim-like) camera
    cam = robot.camera_model()
    # render the floor board through the *rectified* default camera: inverse of H_img2ground
    x0, y0, sq = 0.20, -0.093, 0.031
    g = ground_grid(5, 7, sq, x0, y0).reshape(-1, 2)
    uv, ok = cam.ground_to_image(g)
    assert ok.all()
    # synthesize a rectified image with a checkerboard whose inner corners land on uv
    px = 40
    board = np.full((8 * px, 10 * px), 255, np.uint8)
    for j in range(6):
        for i in range(8):
            if (i + j) % 2 == 0:
                board[(j + 1) * px:(j + 2) * px, (i + 1) * px:(i + 2) * px] = 0
    src = np.array([[(i + 2) * px, (j + 2) * px] for j in range(5) for i in range(7)], np.float32)
    # inner corner (row j, col i): ground x grows with j (away from robot), y grows with i (to the left)
    H, _ = cv2.findHomography(src, uv.astype(np.float32))
    rect = cv2.warpPerspective(board, H, (640, 480), borderValue=110)
    rect = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
    # robot default camera: rectifier is a mild pinhole undistortion; feed the image as "raw" with an
    # identity rectifier so the synthetic geometry is exact
    robot.camera = cam.to_dict()
    robot.camera["rectifier"] = None
    vis, mean_err, max_err = compute_extrinsics(rect, robot, 7, 5, sq, x0, y0)
    assert max_err < 0.002 and robot.calibrated["extrinsics"]
    new = robot.camera_model()
    back, _ = new.image_to_ground(uv)
    assert np.abs(back - g).max() < 0.002


def test_wheel_calibration_math():
    from duckie_yolo.control.kinematics import WheelCalibration
    from duckie_yolo.robot.calibrate_wheels import apply_straight, omega_gain_from

    cal = apply_straight(WheelCalibration(), v=0.25, duration=3.0, drift_cm=6.0, distance_m=0.6)
    assert cal.trim < 0  # drifted left -> less right-wheel
    assert abs(cal.gain - 0.75 / 0.6) < 1e-9
    assert abs(omega_gain_from(np.degrees(3.0), 2.0, 3.0) - 2.0) < 1e-9  # turned half as much


def test_view_selector():
    from duckie_yolo.robot.calibrate_intrinsics import ViewSelector

    sel = ViewSelector(min_move=40, hold_s=3)
    c = np.array([[100.0, 100.0], [140, 100]])
    assert sel.accept(c, now=0.0)
    assert not sel.accept(c + 10, now=1.0)  # barely moved
    assert sel.accept(c + 60, now=1.5)      # moved enough
    assert sel.accept(c + 60, now=5.0)      # held still long enough
