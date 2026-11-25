"""Closed-loop wheel control inside robot_io: start power, start kick and yaw-rate PID (gyro).

Commands arrive as wheel duties (feed-forward from the (v, omega) kinematics) plus, optionally,
the yaw rate they are meant to produce (``omega_ref``). A 100 Hz loop then
  1. PID: compares omega_ref with the gyro and adds +-u to the right/left wheel, so "straight"
     really is straight and turns really turn as much as asked (motor mismatch, battery level,
     floor friction and low-power stalls are corrected automatically);
  2. start power: every moving wheel gets at least ``min_left`` / ``min_right`` duty, so slow
     commands do not stall the motors;
  3. start kick: a short burst of ``kick`` duty when a wheel starts from standstill (static friction).
Without a gyro (or with the PID off) 2 and 3 still apply.

Self-tests that need the motors (robot on the floor):
  * imu_calibrate: robot still -> gyro bias; short left spin -> gyro sign (+ checks the gyro sees turns)
  * find_start_power: pivots on each wheel with slowly rising power until the gyro sees motion
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class MotorTuning:
    min_left: float = 0.0  # duty floor for a moving wheel (0..0.8)
    min_right: float = 0.0
    kick: float = 0.0  # duty during a start kick (0 = off)
    kick_ms: float = 120.0
    swap_sides: bool = False  # left/right motors wired the other way round
    pid_enabled: bool = False
    kp: float = 0.08  # duty per rad/s of yaw-rate error (added to the right wheel, removed from the left)
    ki: float = 1.0
    kd: float = 0.0
    i_max: float = 0.25  # anti-windup limit of the integral term (duty)
    u_max: float = 0.5  # limit of the whole correction (duty)
    imu_sign: float = 1.0  # +1 / -1: makes a left (counter-clockwise) turn positive
    gyro_bias: float = 0.0  # rad/s, subtracted before the sign
    imu_calibrated: bool = False

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MotorTuning":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def apply_floor(d: float, floor: float, deadzone: float = 0.01) -> float:
    """Map |duty| in (deadzone, 1] onto [floor, 1] (same sign); tiny commands stay off."""
    if not math.isfinite(d) or abs(d) < deadzone:
        return 0.0
    floor = min(max(floor, 0.0), 0.95)
    return math.copysign(floor + (1.0 - floor) * min(abs(d), 1.0), d)


class DriveController:
    def __init__(self, wheels, imu=None, tuning: Optional[MotorTuning] = None, rate_hz: float = 100.0,
                 clock=time.monotonic, sleep=time.sleep, start_thread: bool = True):
        self.wheels = wheels
        self.imu = imu
        self.t = tuning or MotorTuning()
        self.dt = 1.0 / rate_hz
        self._clock, self._sleep = clock, sleep
        self._lock = threading.RLock()
        self._cmd: Tuple[float, float] = (0.0, 0.0)
        self._wref: Optional[float] = None
        self._integral = 0.0
        self._prev_wz: Optional[float] = None
        self._prev_applied = (0.0, 0.0)
        self._kick_until = [0.0, 0.0]
        self._t_last = clock()
        self.wz = 0.0  # last corrected yaw rate (rad/s)
        self.yaw = 0.0  # integrated heading (rad), for "how much did it turn" checks
        self.u = 0.0
        self.applied = (0.0, 0.0)
        self.imu_error: Optional[str] = None
        self.task: Optional[Dict[str, Any]] = None  # running / finished self-test
        self._busy = False  # a self-test owns the motors
        self._running = True
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="drive")
            self._thread.start()

    # ------------------------------------------------------------------ commands
    def command(self, left: float, right: float, omega_ref: Optional[float] = None) -> Tuple[float, float]:
        with self._lock:
            if self._busy:
                return self.applied
            new = (float(left), float(right))
            if new == (0.0, 0.0):
                return self._stop_locked()
            if omega_ref is None or (self._wref is not None and abs(omega_ref - self._wref) > 1.5):
                self._integral = 0.0  # big change of target (or open loop): start the integral afresh
            self._cmd, self._wref = new, (None if omega_ref is None else float(omega_ref))
            return self._apply_locked(self._clock(), read_imu=False)

    def stop(self, force: bool = False) -> None:
        with self._lock:
            if self._busy and not force:
                return  # a self-test is running; it stops the wheels itself
            self._stop_locked()

    def _stop_locked(self) -> Tuple[float, float]:
        self._cmd, self._wref, self._integral, self.u = (0.0, 0.0), None, 0.0, 0.0
        self.wheels.stop()
        self.applied = self._prev_applied = (0.0, 0.0)
        return self.applied

    def configure(self, d: Dict[str, Any]) -> MotorTuning:
        with self._lock:
            merged = self.t.to_dict()
            merged.update({k: v for k, v in (d or {}).items() if k in merged})
            self.t = MotorTuning.from_dict(merged)
            self._integral = 0.0
            return self.t

    # ------------------------------------------------------------------ control loop
    def _read_wz(self) -> Optional[float]:
        if self.imu is None:
            return None
        try:
            raw = self.imu.read_wz()
            self.imu_error = None
        except Exception as e:  # a flaky I2C read must not stop the robot loop
            self.imu_error = str(e)
            return None
        return self.t.imu_sign * (raw - self.t.gyro_bias)

    def _apply_locked(self, now: float, read_imu: bool = True) -> Tuple[float, float]:
        t = self.t
        left, right = self._cmd
        dt = max(1e-3, min(0.1, now - self._t_last))
        self._t_last = now
        wz = self._read_wz() if read_imu else None
        if wz is not None:
            self.wz = wz
            self.yaw += wz * dt
        if (left, right) == (0.0, 0.0):
            return self.applied
        u = 0.0
        if t.pid_enabled and self._wref is not None and self.imu is not None and t.imu_calibrated:
            if wz is not None:
                e = self._wref - wz
                self._integral = max(-t.i_max, min(t.i_max, self._integral + t.ki * e * dt))
                d = 0.0 if self._prev_wz is None else -(wz - self._prev_wz) / dt  # on measurement: no kick
                self._prev_wz = wz
                self.u = max(-t.u_max, min(t.u_max, t.kp * e + self._integral + t.kd * d))
            u = self.u
        out = []
        for i, (d, floor) in enumerate(((left - u, t.min_left), (right + u, t.min_right))):
            d = apply_floor(d, floor)
            prev = self._prev_applied[i]
            if t.kick > 0 and d != 0.0 and (prev == 0.0 or math.copysign(1, prev) != math.copysign(1, d)):
                self._kick_until[i] = now + t.kick_ms / 1000.0
            if d != 0.0 and now < self._kick_until[i]:
                d = math.copysign(max(abs(d), t.kick), d)
            out.append(d)
        l, r = out
        self._prev_applied = (l, r)
        if t.swap_sides:
            l, r = r, l
        self.wheels.set(l, r)
        self.applied = (l, r)
        return self.applied

    def _loop(self) -> None:
        next_t = self._clock()
        while self._running:
            with self._lock:
                if not self._busy:
                    now = self._clock()
                    if self._cmd == (0.0, 0.0):
                        dt = max(1e-3, min(0.1, now - self._t_last))
                        self._t_last = now
                        wz = self._read_wz()
                        if wz is not None:
                            self.wz = wz
                            self.yaw += wz * dt
                    else:
                        self._apply_locked(now)
            next_t += self.dt
            delay = next_t - self._clock()
            if delay > 0:
                self._sleep(delay)
            else:
                next_t = self._clock()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            self._busy = False
            self._stop_locked()

    # ------------------------------------------------------------------ self-tests
    def start_task(self, name: str) -> Dict[str, Any]:
        fn = {"imu_calibrate": self._task_imu_calibrate, "find_start_power": self._task_find_start_power}.get(name)
        if fn is None:
            raise ValueError(f"unknown task {name!r}")
        if self.imu is None:
            raise RuntimeError("no IMU found: this test needs the gyro")
        with self._lock:
            if self._busy:
                raise RuntimeError("another motor test is running")
            self._stop_locked()
            self._busy = True
            self.task = {"name": name, "state": "running", "result": None, "error": None, "t": time.time()}
        threading.Thread(target=self._run_task, args=(fn,), daemon=True, name="drive-task").start()
        return dict(self.task)

    def _run_task(self, fn) -> None:
        try:
            result = fn()
            self.task.update(state="done", result=result)
        except Exception as e:
            self.task.update(state="failed", error=str(e))
        finally:
            with self._lock:
                self.wheels.stop()
                self.applied = self._prev_applied = (0.0, 0.0)
                self._busy = False

    def _raw_wz_samples(self, seconds: float):
        out = []
        t_end = self._clock() + seconds
        while self._clock() < t_end:
            out.append(self.imu.read_wz())
            self._sleep(self.dt)
        return out

    def _task_imu_calibrate(self) -> Dict[str, Any]:
        self.wheels.stop()
        self._sleep(0.3)
        still = self._raw_wz_samples(1.5)
        bias = sum(still) / len(still)
        noise = (sum((s - bias) ** 2 for s in still) / len(still)) ** 0.5
        # commanded LEFT (counter-clockwise) spin: right wheel forward, left backward
        p = max(0.35, self.t.min_left + 0.1, self.t.min_right + 0.1)
        l, r = (-p, p) if not self.t.swap_sides else (p, -p)
        self.wheels.set(l, r)
        spin = self._raw_wz_samples(0.6)
        self.wheels.stop()
        self._sleep(0.3)
        mean = sum(spin) / len(spin) - bias
        if abs(mean) < 0.4:
            raise RuntimeError(f"the gyro hardly saw the spin ({math.degrees(mean):.0f} deg/s): is the robot on the "
                               "floor, are the wheels turning, is the IMU mounted flat?")
        with self._lock:
            self.t.gyro_bias = bias
            self.t.imu_sign = 1.0 if mean > 0 else -1.0
            self.t.imu_calibrated = True
            self.yaw = 0.0
        return {"gyro_bias": bias, "noise_dps": math.degrees(noise), "imu_sign": self.t.imu_sign,
                "spin_dps": math.degrees(abs(mean)), "turned_deg": math.degrees(abs(mean) * 0.6)}

    def _task_find_start_power(self) -> Dict[str, Any]:
        if not self.t.imu_calibrated:
            raise RuntimeError("calibrate the IMU first")
        found = {}
        for side in ("left", "right"):
            self.wheels.stop()
            self._sleep(0.5)
            hits, duty = 0, 0.0
            while duty < 0.8:
                duty += 0.02
                l, r = (duty, 0.0) if side == "left" else (0.0, duty)
                if self.t.swap_sides:
                    l, r = r, l
                self.wheels.set(l, r)
                self._sleep(0.12)
                wz = self.t.imu_sign * (self.imu.read_wz() - self.t.gyro_bias)
                hits = hits + 1 if abs(wz) > 0.35 else 0
                if hits >= 2:
                    break
            self.wheels.stop()
            if hits < 2:
                raise RuntimeError(f"the {side} wheel never moved the robot, even at {duty:.0%} power")
            found[side] = round(min(0.8, duty + 0.03), 3)  # small margin above the start point
        with self._lock:
            self.t.min_left, self.t.min_right = found["left"], found["right"]
        return {"min_left": found["left"], "min_right": found["right"]}

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        return {
            "imu": getattr(self.imu, "chip", None) if self.imu is not None else None,
            "imu_error": self.imu_error,
            "wz": round(self.wz, 4),
            "yaw": round(self.yaw, 4),
            "wref": self._wref,
            "u": round(self.u, 4),
            "applied": [round(a, 3) for a in self.applied],
            "busy": self._busy,
            "task": self.task,
        }
