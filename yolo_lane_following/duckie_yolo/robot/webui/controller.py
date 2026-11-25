"""RobotController: the web UI's single connection to robot_io and its control loop.

Each loop iteration: take the latest camera frame -> (model inference) -> choose the wheel
command -> send it (which returns the next frame) -> record -> publish the view.

Who drives, highest priority first:
  1. calibration motion (timed straight run / spin started from the Calibrate tab)
  2. teleop: a drive command from the page within the last TELEOP_TIMEOUT s (hold-to-drive)
  3. autodrive: the model's command, only while the page sends heartbeats (HEARTBEAT_TIMEOUT)
  4. otherwise zero
STOP cancels 1-3 and switches autodrive off. robot_io's own watchdog and duty cap stay in force.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..drive import MotorTuning
from ..mjpeg import FrameBuffer
from ..real_env import RealRobotEnv
from ..robot_config import RobotConfig
from .models import ModelManager
from .recorder import Recorder
from .util import disk_free_gb

TELEOP_TIMEOUT = 0.3
STEP_MAX_AGE = 0.5  # s: a model result older than this never drives the robot
HEARTBEAT_TIMEOUT = 1.0
VIEWS = ("raw", "rect", "model")


def _text(img: np.ndarray, lines, org=(8, 22), scale=0.55) -> np.ndarray:
    x, y = org
    for line in lines:
        cv2.putText(img, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(36 * scale + 4)
    return img


class RobotController:
    def __init__(
        self,
        robot_host: str = "localhost",
        robot_port: int = 5555,
        robot_config: str = "configs/robot_mybot.yaml",
        models: Optional[ModelManager] = None,
        recorder: Optional[Recorder] = None,
        encoding: str = "raw",
        stream_fps: float = 30.0,
        teleop_timeout: float = TELEOP_TIMEOUT,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
    ):
        self.host, self.port, self.encoding = robot_host, robot_port, encoding
        self.robot_config_path = robot_config
        self.models = models or ModelManager("models")
        self.recorder = recorder or Recorder("data/real")
        self.frames = FrameBuffer(quality=75)
        self.stream_fps = stream_fps
        self.teleop_timeout = teleop_timeout
        self.heartbeat_timeout = heartbeat_timeout

        self._lock = threading.RLock()
        self.robot = RobotConfig.load(robot_config)
        self.camera = self.robot.camera_model()
        self._env: Optional[RealRobotEnv] = None
        self.connected = False
        self.robot_error: Optional[str] = None

        self.view = "rect"
        self.viewers = 0
        self._teleop = (0.0, 0.0)
        self._teleop_t = -1e9
        self.autodrive = False
        self.max_speed = 0.2
        self._hb_t = -1e9
        self._motion: Optional[Tuple[float, float, float, float]] = None  # v, w, t_start, t_end
        self.overlay_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None  # e.g. checkerboard corners

        self.drive: Dict[str, Any] = {}  # robot_io drive controller status (gyro, PID, self-tests)
        self.trace: "collections.deque" = collections.deque(maxlen=600)  # (t, wref, wz, left, right)
        self.motion_result: Optional[Dict[str, Any]] = None
        self._motion_yaw0: Optional[float] = None
        self._motion_kind: Optional[Tuple[float, float, float]] = None
        self._rpcs: "queue.Queue" = queue.Queue()
        self._task_state: Optional[str] = None
        self.motors_supported: Optional[bool] = None  # robot_io understands "configure"

        self._raw: Optional[np.ndarray] = None  # latest raw BGR frame
        self._raw_seq = 0
        self.cmd = (0.0, 0.0)
        self.source = "idle"
        self.hz = 0.0
        self.model_hz = 0.0
        self.infer_ms = 0.0
        self._raw_rgb: Optional[np.ndarray] = None
        self._frame_cond = threading.Condition()
        self._step = None  # newest AgentStep from the inference thread
        self._step_t = 0.0
        self.event: Optional[str] = None
        self._event_t = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "RobotController":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="control-loop")
        self._thread.start()
        threading.Thread(target=self._infer_loop, daemon=True, name="inference").start()
        threading.Thread(target=self._render_loop, daemon=True, name="render").start()
        return self

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.recorder.stop()

    # ------------------------------------------------------------------ commands from the page
    def teleop(self, v: float, w: float) -> None:
        with self._lock:
            self._teleop = (float(np.clip(v, -0.6, 0.6)), float(np.clip(w, -8.0, 8.0)))
            self._teleop_t = time.monotonic()

    def stop(self, reason: str = "STOP pressed") -> None:
        with self._lock:
            self._teleop, self._teleop_t = (0.0, 0.0), -1e9
            self._motion = None
            if self.autodrive:
                self.autodrive = False
            self._set_event(reason)

    def heartbeat(self) -> None:
        with self._lock:
            self._hb_t = time.monotonic()

    def set_autodrive(self, enabled: bool, max_speed: Optional[float] = None) -> None:
        with self._lock:
            if max_speed is not None:
                self.max_speed = float(np.clip(max_speed, 0.0, 0.5))
            if enabled and self.models.state != "running":
                raise ValueError("start a model first")
            self.autodrive = bool(enabled)
            if enabled:
                self._hb_t = time.monotonic()
                self.models.agent.reset()  # fresh FSM state for a new run

    def set_view(self, view: str) -> None:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.view = view

    def start_motion(self, v: float, w: float, duration: float, delay: float = 2.0) -> None:
        """Timed open-loop motion for wheel calibration (starts after `delay` s)."""
        with self._lock:
            if self.autodrive:
                raise ValueError("switch autodrive off first")
            t0 = time.monotonic() + delay
            self._motion = (float(v), float(w), t0, t0 + float(duration))
            self._motion_kind = (float(v), float(w), float(duration))
            self._motion_yaw0 = None
            self.motion_result = None

    # ------------------------------------------------------------------ helpers for other modules
    def latest_raw(self) -> Tuple[Optional[np.ndarray], int]:
        with self._frame_cond:
            return self._raw, self._raw_seq

    def reload_robot_config(self) -> None:
        robot = RobotConfig.load(self.robot_config_path)
        cam = robot.camera_model()
        with self._lock:
            self.robot, self.camera = robot, cam
            if self._env is not None:
                self._env.cal = robot.wheels
        self.models.set_camera(cam, robot.effective_actuation())

    def robot_rpc(self, msg: dict, timeout: float = 3.0) -> Optional[dict]:
        """Run a JSON-only robot_io command from an HTTP thread (the loop thread owns the socket)."""
        if self._env is None:
            raise RuntimeError("robot_io is not connected")
        done, box = threading.Event(), {}
        self._rpcs.put((msg, done, box))
        if not done.wait(timeout):
            raise RuntimeError("robot_io did not answer in time")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("reply")

    def motor_tuning(self) -> MotorTuning:
        return MotorTuning.from_dict(self.robot.motors)

    def set_motor_tuning(self, changes: Dict[str, Any]) -> MotorTuning:
        """Merge, save to the robot config and send to robot_io."""
        robot = RobotConfig.load(self.robot_config_path)
        t = MotorTuning.from_dict(dict(MotorTuning.from_dict(robot.motors).to_dict(), **changes))
        robot.motors = t.to_dict()
        robot.save(self.robot_config_path)
        from .util import own

        own(self.robot_config_path)
        self.reload_robot_config()
        reply = self.robot_rpc({"command": "configure", "tuning": robot.motors})
        if reply is None:
            raise RuntimeError("robot_io does not support motor tuning yet: restart it after `git pull` "
                               "(sudo systemctl restart duckie-robot-io)")
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "robot_io refused the tuning"))
        return t

    def _sync_tuning_from_robot(self) -> None:
        """A self-test changed the tuning inside robot_io: copy it into the robot config file."""
        reply = self._env.rpc({"command": "configure", "tuning": {}}) if self._env is not None else None
        if not reply or not reply.get("ok"):
            return
        robot = RobotConfig.load(self.robot_config_path)
        robot.motors = reply["tuning"]
        robot.save(self.robot_config_path)
        from .util import own

        own(self.robot_config_path)
        self.reload_robot_config()

    def _process_rpcs(self) -> None:
        while True:
            try:
                msg, done, box = self._rpcs.get_nowait()
            except queue.Empty:
                return
            try:
                box["reply"] = self._env.rpc(msg, timeout_ms=1500)
            except Exception as e:
                box["error"] = str(e)
            done.set()

    def trace_samples(self, seconds: float = 8.0) -> List[List[float]]:
        now = time.monotonic()
        return [[round(t - now, 3), wref, wz, l, r] for (t, wref, wz, l, r) in list(self.trace) if now - t <= seconds]

    def rectify(self, bgr: np.ndarray) -> np.ndarray:
        return self.camera.rectify(bgr)

    def viewer_enter(self) -> None:
        with self._lock:
            self.viewers += 1

    def viewer_exit(self) -> None:
        with self._lock:
            self.viewers = max(0, self.viewers - 1)

    def _set_event(self, text: str) -> None:
        self.event, self._event_t = text, time.time()
        print(f"[webui] {text}", flush=True)

    # ------------------------------------------------------------------ control loop
    def _choose(self, now: float, step) -> Tuple[Tuple[float, float], str]:
        with self._lock:
            if self._motion is not None:
                v, w, t0, t1 = self._motion
                if now < t0:
                    return (0.0, 0.0), "calib-wait"
                if now < t1:
                    return (v, w), "calib"
                self._motion = None
            if now - self._teleop_t < self.teleop_timeout:
                return self._teleop, "teleop"
            if self.autodrive:
                if now - self._hb_t > self.heartbeat_timeout:
                    self.autodrive = False
                    self._set_event("autodrive stopped: lost contact with the page")
                elif self.models.state not in ("running", "loading"):
                    self.autodrive = False
                    self._set_event("autodrive stopped: model not running")
                elif step is not None:
                    v = float(np.clip(step.action[0], -self.max_speed, self.max_speed))
                    return (v, float(step.action[1])), "auto"
                else:
                    return (0.0, 0.0), "auto"
            return (0.0, 0.0), "idle"

    def _connect(self) -> Optional[np.ndarray]:
        env = None
        try:
            env = RealRobotEnv(self.host, self.port, self.robot.wheels, encoding=self.encoding, timeout_ms=1500)
            obs = env.reset()
        except Exception as e:
            if env is not None:
                env.sock.close(0)
            if self.connected or self.robot_error is None:
                print(f"[webui] robot_io not reachable at {self.host}:{self.port}: {e}", flush=True)
            self.connected, self.robot_error = False, f"robot_io not reachable at {self.host}:{self.port} ({e})"
            return None
        with self._lock:
            self._env = env
        self.connected, self.robot_error = True, None
        print(f"[webui] connected to robot_io at {self.host}:{self.port}", flush=True)
        try:
            reply = env.rpc({"command": "configure", "tuning": dict(self.robot.motors or {})}, timeout_ms=1000)
        except Exception:
            reply = None
        self.motors_supported = bool(reply and reply.get("ok"))
        if not self.motors_supported:
            print("[webui] robot_io has no motor tuning (old version?): restart it after git pull", flush=True)
            obs = env.reset()  # the timed-out request reconnected the socket
        return obs

    def _disconnect(self, err: Exception) -> None:
        with self._lock:
            env, self._env = self._env, None
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        self.connected, self.robot_error = False, f"lost robot_io: {err}"
        print(f"[webui] lost robot_io: {err}", flush=True)

    def _placeholder(self, text: str) -> None:
        img = np.full((480, 640, 3), 40, np.uint8)
        _text(img, ["No camera", text[:80]], org=(20, 220), scale=0.7)
        self.frames.publish(img)

    def _loop(self) -> None:
        """Camera-rate loop (~30 Hz): frame in, wheel command out, recording. Never waits for the model."""
        obs = None
        t_prev = time.monotonic()
        while self._running:
            if self._env is None:
                obs = self._connect()
                if obs is None:
                    self._placeholder(self.robot_error or "")
                    time.sleep(1.0)
                    continue
            now = time.monotonic()
            raw = np.ascontiguousarray(obs[..., ::-1])  # robot_io delivers BGR; RealRobotEnv hands out RGB
            with self._frame_cond:
                self._raw, self._raw_rgb, self._raw_seq = raw, obs, self._raw_seq + 1
                self._frame_cond.notify_all()

            # newest model result; a result older than STEP_MAX_AGE never drives the robot
            step = self._step if self._step is not None and now - self._step_t < STEP_MAX_AGE else None
            cmd, source = self._choose(now, step)
            try:
                self._process_rpcs()
                obs, _, _, _ = self._env.step(cmd)
            except Exception as e:
                self._disconnect(e)
                continue
            self.cmd, self.source = cmd, source
            self._after_step(now, cmd, source)

            if self.recorder.active and self.recorder.due(time.time()):
                self.recorder.maybe_save(time.time(), raw, self.camera.rectify(raw))

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                self.hz = 0.9 * self.hz + 0.1 / dt if self.hz else 1.0 / dt
        # leaving: release the wheels
        with self._lock:
            env, self._env = self._env, None
        if env is not None:
            env.close()

    def _infer_loop(self) -> None:
        """Model inference on the newest camera frame, as fast as the model allows."""
        last_seq = -1
        t_prev = None
        while self._running:
            agent = self.models.agent
            if agent is None:
                self._step, self.model_hz, t_prev = None, 0.0, None
                time.sleep(0.05)
                continue
            with self._frame_cond:
                if self._raw_seq == last_seq:
                    self._frame_cond.wait(0.2)
                rgb, seq = self._raw_rgb, self._raw_seq
            if rgb is None or seq == last_seq:
                continue
            last_seq = seq
            try:
                step = agent.act(rgb)
            except Exception as e:
                self.models.fail(f"inference failed: {type(e).__name__}: {e}")
                self._set_event(f"model stopped: {e}")
                continue
            if self.models.agent is not agent:
                continue  # model stopped / replaced meanwhile
            now = time.monotonic()
            self._step, self._step_t = step, now
            self.infer_ms = 0.8 * self.infer_ms + 0.2 * 1000.0 * step.timings["segment"] if self.infer_ms else \
                1000.0 * step.timings["segment"]
            if t_prev is not None and now > t_prev:
                self.model_hz = 0.8 * self.model_hz + 0.2 / (now - t_prev) if self.model_hz else 1.0 / (now - t_prev)
            t_prev = now

    def _render_loop(self) -> None:
        """Draws the selected view and JPEG-encodes it for the browser (only while someone watches)."""
        last_seq = -1
        while self._running:
            if self.viewers <= 0 or self._env is None:
                time.sleep(0.1)
                continue
            t0 = time.monotonic()
            raw, seq = self._raw, self._raw_seq
            if raw is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                self.frames.publish(self._render(raw, self._step))
            except Exception as e:  # a drawing bug must never stop anything else
                print(f"[webui] render failed: {e}", flush=True)
                time.sleep(0.5)
            rest = 1.0 / self.stream_fps - (time.monotonic() - t0)
            if rest > 0:
                time.sleep(rest)

    def _after_step(self, now: float, cmd: Tuple[float, float], source: str) -> None:
        d = self._env.last_md.get("drive") if self._env is not None else None
        if not d:
            return
        self.drive = d
        app = d.get("applied") or [0.0, 0.0]
        self.trace.append((now, cmd[1] if cmd != (0.0, 0.0) else 0.0, float(d.get("wz", 0.0)), app[0], app[1]))
        # heading change during a calibration run ("drive straight" should end near 0 deg)
        if source == "calib" and self._motion_yaw0 is None:
            self._motion_yaw0 = float(d.get("yaw", 0.0))
        elif source != "calib" and self._motion_yaw0 is not None and self._motion is None:
            v, w, dur = self._motion_kind or (0.0, 0.0, 0.0)
            turned = np.degrees(float(d.get("yaw", 0.0)) - self._motion_yaw0)
            self.motion_result = {"v": v, "w": w, "duration": dur, "turned_deg": round(float(turned), 1),
                                  "expected_deg": round(float(np.degrees(w * dur)), 1), "t": time.time(),
                                  "pid": bool(self.robot.pid_active())}
            self._motion_yaw0 = None
        task = d.get("task") or {}
        state = task.get("state")
        if state != self._task_state:
            if self._task_state == "running" and state == "done":
                self._sync_tuning_from_robot()
            self._task_state = state

    def _render(self, raw: np.ndarray, step) -> np.ndarray:
        view = self.view
        overlay = self.overlay_fn
        if overlay is not None:
            return overlay(raw.copy())
        if view == "raw":
            return raw
        if view == "model":
            if step is None:
                return _text(self.camera.rectify(raw), [f"no model running ({self.models.state})"])
            from ...viz import compose

            e = step.estimate
            lines = [
                f"{step.state.value}  v={self.cmd[0]:.2f} w={self.cmd[1]:+.2f}  [{self.source}]",
                f"d={e.d:+.3f} m  phi={np.degrees(e.phi):+.1f} deg  conf={e.confidence:.2f}  src={e.source}",
                f"camera {self.hz:.0f} Hz  model {self.model_hz:.1f} Hz  inference {self.infer_ms:.0f} ms",
            ]
            return compose(step.image, step.segmentation, e, self.camera, lines)
        return self.camera.rectify(raw)

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            motion = None
            if self._motion is not None:
                v, w, t0, t1 = self._motion
                motion = {"v": v, "w": w, "starts_in": max(0.0, t0 - now), "remaining": max(0.0, t1 - max(now, t0))}
            r = self.robot
            return {
                "robot": {"connected": self.connected, "error": self.robot_error, "host": f"{self.host}:{self.port}"},
                "loop": {"hz": round(self.hz, 1), "model_hz": round(self.model_hz, 1), "infer_ms": round(self.infer_ms, 1),
                         "cmd": [round(c, 3) for c in self.cmd],
                         "source": self.source},
                "view": self.view,
                "viewers": self.viewers,
                "autodrive": {"enabled": self.autodrive, "max_speed": self.max_speed},
                "motion": motion,
                "model": self.models.status(),
                "record": self.recorder.status(),
                "calibration": {
                    "file": self.robot_config_path,
                    "calibrated": dict(r.calibrated),
                    "wheels": {"gain": round(r.wheels.gain, 4), "trim": round(r.wheels.trim, 4)},
                    "omega_gain": round(r.actuation.omega_gain, 4),
                },
                "motors": {
                    "tuning": MotorTuning.from_dict(r.motors).to_dict(),
                    "supported": self.motors_supported,
                    "drive": self.drive,
                    "motion_result": self.motion_result,
                    "pid_active": r.pid_active(),
                },
                "event": self.event if self.event and time.time() - self._event_t < 15 else None,
                "disk_free_gb": round(disk_free_gb(self.recorder.root), 1),
            }
