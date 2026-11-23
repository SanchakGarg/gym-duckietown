"""Differential-drive kinematics: (v, omega) commands -> left/right wheel duty cycles.

This is exactly the conversion done by ``gym_duckietown.envs.DuckietownEnv.step`` so that a
command means the same thing in simulation and on the real robot. On the robot, ``gain``
and ``trim`` come from the wheel calibration (robot/calibrate_wheels.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class WheelCalibration:
    gain: float = 1.0
    trim: float = 0.0  # > 0 makes the robot turn more to the left
    radius: float = 0.0318  # wheel radius (m)
    k: float = 27.0  # motor constant
    limit: float = 1.0  # duty cycle saturation
    baseline: float = 0.102  # distance between the wheels (m)


def twist_to_wheels(v: float, omega: float, cal: WheelCalibration = WheelCalibration()) -> Tuple[float, float]:
    k_r_inv = (cal.gain + cal.trim) / cal.k
    k_l_inv = (cal.gain - cal.trim) / cal.k
    omega_r = (v + 0.5 * omega * cal.baseline) / cal.radius
    omega_l = (v - 0.5 * omega * cal.baseline) / cal.radius
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv
    u_r = max(min(u_r, cal.limit), -cal.limit)
    u_l = max(min(u_l, cal.limit), -cal.limit)
    return u_l, u_r
