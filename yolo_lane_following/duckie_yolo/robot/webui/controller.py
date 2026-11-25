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

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from ..mjpeg import FrameBuffer
from ..real_env import RealRobotEnv
from ..robot_config import RobotConfig
from .models import ModelManager
from .recorder import Recorder
from .util import disk_free_gb

TELEOP_TIMEOUT = 0.3
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
        stream_fps: float = 15.0,
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

        self._raw: Optional[np.ndarray] = None  # latest raw BGR frame
        self._raw_seq = 0
        self.cmd = (0.0, 0.0)
        self.source = "idle"
        self.hz = 0.0
        self.infer_ms = 0.0
        self.event: Optional[str] = None
        self._event_t = 0.0
        self._t_pub = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "RobotController":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="control-loop")
        self._thread.start()
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

    # ------------------------------------------------------------------ helpers for other modules
    def latest_raw(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._raw, self._raw_seq

    def reload_robot_config(self) -> None:
        robot = RobotConfig.load(self.robot_config_path)
        cam = robot.camera_model()
        with self._lock:
            self.robot, self.camera = robot, cam
            if self._env is not None:
                self._env.cal = robot.wheels
        self.models.set_camera(cam, robot.actuation)

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
            with self._lock:
                self._raw, self._raw_seq = raw, self._raw_seq + 1

            step = None
            agent = self.models.agent
            if agent is not None:
                try:
                    step = agent.act(obs)
                    self.infer_ms = 0.8 * self.infer_ms + 0.2 * 1000.0 * step.timings["segment"]
                except Exception as e:
                    self.models.fail(f"inference failed: {type(e).__name__}: {e}")
                    self._set_event(f"model stopped: {e}")

            rect_cache = []

            def rect() -> np.ndarray:
                if not rect_cache:
                    rect_cache.append(np.ascontiguousarray(step.image[..., ::-1]) if step is not None else self.camera.rectify(raw))
                return rect_cache[0]

            cmd, source = self._choose(now, step)
            try:
                obs, _, _, _ = self._env.step(cmd)
            except Exception as e:
                self._disconnect(e)
                continue
            self.cmd, self.source = cmd, source

            if self.recorder.active:
                self.recorder.maybe_save(time.time(), raw, rect())
            if self.viewers > 0 and now - self._t_pub >= 1.0 / self.stream_fps:
                self._t_pub = now
                try:
                    self.frames.publish(self._render(raw, rect, step))
                except Exception as e:  # a drawing bug must never stop the robot loop
                    print(f"[webui] render failed: {e}", flush=True)

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                self.hz = 0.9 * self.hz + 0.1 / dt if self.hz else 1.0 / dt
        # leaving: release the wheels
        with self._lock:
            env, self._env = self._env, None
        if env is not None:
            env.close()

    def _render(self, raw: np.ndarray, rect: Callable[[], np.ndarray], step) -> np.ndarray:
        view = self.view
        if self.overlay_fn is not None:
            return self.overlay_fn(raw.copy())
        if view == "raw":
            return raw
        if view == "model":
            if step is None:
                return _text(rect().copy(), [f"no model running ({self.models.state})"])
            from ...viz import compose

            e = step.estimate
            lines = [
                f"{step.state.value}  v={self.cmd[0]:.2f} w={self.cmd[1]:+.2f}  [{self.source}]",
                f"d={e.d:+.3f} m  phi={np.degrees(e.phi):+.1f} deg  conf={e.confidence:.2f}  src={e.source}",
                f"{self.hz:.1f} Hz  inference {self.infer_ms:.0f} ms",
            ]
            return compose(step.image, step.segmentation, e, self.camera, lines)
        return rect()

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
                "loop": {"hz": round(self.hz, 1), "infer_ms": round(self.infer_ms, 1), "cmd": [round(c, 3) for c in self.cmd],
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
                "event": self.event if self.event and time.time() - self._event_t < 15 else None,
                "disk_free_gb": round(disk_free_gb(self.recorder.root), 1),
            }
