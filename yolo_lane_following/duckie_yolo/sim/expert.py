"""Privileged pure-pursuit expert (uses the simulator's ground-truth lane curves).

Adapted from ``test_segmentation.py::PurePursuitPolicy`` in gym-duckietown (that file runs a
script at import time, so it cannot be imported). Used to drive the robot while recording
the auto-labelled dataset and as the upper-bound baseline in evaluation.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class PurePursuitExpert:
    def __init__(self, env, ref_velocity: float = 0.45, following_distance: float = 0.24, omega_gain: float = 7.0):
        self.env = env
        self.ref_velocity = ref_velocity
        self.following_distance = following_distance
        self.omega_gain = omega_gain

    def _projected(self, lookup_distance: float):
        u = self.env.unwrapped
        closest_point, closest_tangent = u.closest_curve_point(u.cur_pos, u.cur_angle)
        if closest_point is None:
            return None, None, None
        curve_point, curve_tangent = None, None
        for _ in range(10):
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_tangent = u.closest_curve_point(follow_point, u.cur_angle)
            if curve_point is not None and curve_tangent is not None:
                break
            lookup_distance *= 0.5
        if curve_tangent is None:
            return None, None, None
        return float(np.dot(curve_tangent, closest_tangent)), closest_point, curve_point

    def predict(self) -> Tuple[float, float]:
        u = self.env.unwrapped
        proj_angle, _, _ = self._projected(0.3)
        if proj_angle is None:
            return 0.0, 0.0
        scale = 1.0
        i, j = u.get_grid_coords(u.cur_pos)
        tile = u._get_tile(i, j)
        if (tile is not None and "curve" in tile["kind"]) or abs(proj_angle) < 0.92:
            scale = 0.5
        _, closest_point, curve_point = self._projected(self.following_distance)
        if closest_point is None:
            return 0.0, 0.0
        point_vec = curve_point - u.cur_pos
        point_vec /= max(np.linalg.norm(point_vec), 1e-6)
        right_vec = np.array([math.sin(u.cur_angle), 0.0, math.cos(u.cur_angle)])
        omega = -float(np.dot(right_vec, point_vec)) * self.omega_gain
        return self.ref_velocity * scale, omega
