"""Gym-style client for robot_io: the real robot behind the same reset/step API as the simulator.

``step([v, omega])`` uses the same (v, omega) -> wheel duty conversion as
gym_duckietown.envs.DuckietownEnv, so a policy sends identical commands in sim and reality.
"""
from __future__ import annotations

import itertools
from typing import Optional, Tuple

import numpy as np

from ..control.kinematics import WheelCalibration, twist_to_wheels
from .protocol import DEFAULT_PORT, decode_frame


class RealRobotEnv:
    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        calibration: Optional[WheelCalibration] = None,
        encoding: str = "raw",
        jpeg_quality: int = 80,
        timeout_ms: int = 2000,
        wait_new_frame: bool = True,
    ):
        import zmq

        self.zmq = zmq
        self.cal = calibration or WheelCalibration()
        self.encoding = encoding
        self.quality = jpeg_quality
        self.timeout_ms = timeout_ms
        self.wait_new_frame = wait_new_frame
        self.ctx = zmq.Context.instance()
        self.addr = f"tcp://{host}:{port}"
        self.sock = None
        self._ids = itertools.count(1)
        self.last_seq = -1
        self.last_md: dict = {}
        self._connect()

    def _connect(self) -> None:
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(self.zmq.PAIR)
        self.sock.setsockopt(self.zmq.LINGER, 0)
        self.sock.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        self.sock.connect(self.addr)

    def _request(self, msg: dict) -> np.ndarray:
        """Send a command and return the RGB frame of the matching reply."""
        rid = next(self._ids)
        msg = dict(msg, req_id=rid, encoding=self.encoding, quality=self.quality)
        if self.wait_new_frame:
            msg["after_seq"] = self.last_seq
        self.sock.send_json(msg)
        while True:
            try:
                md = self.sock.recv_json()
            except self.zmq.Again:
                self._connect()  # a late reply would otherwise desynchronise the PAIR socket
                raise TimeoutError(f"no reply from robot_io at {self.addr}")
            if "error" in md:
                raise RuntimeError(md["error"])
            payload = self.sock.recv()
            if md.get("req_id") not in (rid, None):
                continue  # stale reply to an earlier (timed-out) request
            self.last_seq = int(md.get("seq", self.last_seq))
            self.last_md = md
            return np.ascontiguousarray(decode_frame(md, payload)[..., ::-1])  # BGR -> RGB

    def reset(self) -> np.ndarray:
        self.last_seq = -1
        return self._request({"command": "reset"})

    def wheels(self, left: float, right: float, omega_ref: Optional[float] = None) -> np.ndarray:
        msg = {"command": "action", "values": [float(left), float(right)]}
        if omega_ref is not None:
            msg["omega_ref"] = float(omega_ref)  # lets robot_io's gyro PID hold this yaw rate
        return self._request(msg)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        v, omega = float(action[0]), float(action[1])
        left, right = twist_to_wheels(v, omega, self.cal)
        obs = self.wheels(left, right, omega)
        return obs, 0.0, False, {"wheels": (left, right), "robot": self.last_md}

    def rpc(self, msg: dict, timeout_ms: Optional[int] = None) -> Optional[dict]:
        """Command answered with JSON only (configure / drive_task). None if robot_io does not answer
        (e.g. an older robot_io without that command)."""
        rid = next(self._ids)
        self.sock.send_json(dict(msg, req_id=rid))
        if timeout_ms is not None:
            self.sock.setsockopt(self.zmq.RCVTIMEO, timeout_ms)
        try:
            while True:
                try:
                    md = self.sock.recv_json()
                except self.zmq.Again:
                    self._connect()
                    return None
                if "dtype" in md or "shape" in md:  # a late frame reply: drop its payload too
                    self.sock.recv()
                    continue
                if md.get("req_id") in (rid, None):
                    return md
        finally:
            if timeout_ms is not None:
                self.sock.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)

    def configure(self, tuning: dict) -> Optional[dict]:
        """Send motor tuning (start power, kick, PID; see robot/drive.py MotorTuning) to robot_io."""
        return self.rpc({"command": "configure", "tuning": dict(tuning or {})})

    def drive_task(self, task: str) -> Optional[dict]:
        """Start a motor self-test on robot_io: "imu_calibrate" or "find_start_power"."""
        return self.rpc({"command": "drive_task", "task": task})

    def stop(self) -> None:
        try:
            self.sock.send_json({"command": "stop", "req_id": next(self._ids)})
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        self.sock.close(0)
