"""High-level behaviour: lane following + stop lines + obstacle stops."""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from ..config import BehaviorConfig
from ..estimation.lane_estimator import LaneEstimate


class State(str, Enum):
    DRIVE = "DRIVE"
    STOP_LINE_WAIT = "STOP_LINE_WAIT"
    CROSS = "CROSS"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    LOST = "LOST"


class VirtualLane:
    """The lane centre line measured at the stop line, carried through the intersection.

    Stored as a point ``p`` and unit direction ``u`` in the robot frame (x forward, y left) and
    moved with dead reckoning from the commanded (physical) v / omega, so the robot keeps
    following the lane's direction across the intersection instead of driving blind.
    """

    def __init__(self, center_poly: Optional[np.ndarray], heading: Optional[float] = None, x_ref: float = 0.0,
                 tau: float = 0.0):
        """Line through the lane centre at x = x_ref. Direction: ``heading`` (e.g. the stop line's
        normal) if given, else the slope of ``center_poly``. ``tau``: actuator lag (s) used when
        dead-reckoning (the robot starts from standstill when crossing)."""
        c0 = float(np.polyval(center_poly, x_ref)) if center_poly is not None else 0.0
        c0 = float(np.clip(c0, -0.12, 0.12))
        if heading is None:
            slope = float(np.polyval(np.polyder(center_poly), x_ref)) if center_poly is not None else 0.0
            heading = math.atan(slope)
        self.p = np.array([x_ref, c0])
        self.u = np.array([math.cos(heading), math.sin(heading)])
        self.tau = tau
        self._v = 0.0
        self._w = 0.0

    def move(self, v: float, omega: float, dt: float) -> None:
        """The robot moved by (v, omega) for dt: express the line in the new robot frame."""
        dth = omega * dt
        # translation along the mid-step heading, then rotation by -dth
        dx = v * dt * math.cos(dth / 2.0)
        dy = v * dt * math.sin(dth / 2.0)
        c, s = math.cos(-dth), math.sin(-dth)
        R = np.array([[c, -s], [s, c]])
        self.p = R @ (self.p - np.array([dx, dy]))
        self.u = R @ self.u

    def target(self, lookahead: float) -> Tuple[float, float]:
        # point on the line at distance `lookahead` from the robot, ahead of it
        b = float(self.p @ self.u)
        cc = float(self.p @ self.p) - lookahead * lookahead
        disc = b * b - cc
        if disc < 0:  # line further than the lookahead: aim at its closest point
            s = -b
        else:
            s = -b + math.sqrt(disc)
        q = self.p + s * self.u
        return float(q[0]), float(q[1])


class BehaviorFSM:
    def __init__(self, cfg: BehaviorConfig, lookahead: float = 0.3):
        self.cfg = cfg
        self.lookahead = lookahead
        self.reset()

    def reset(self) -> None:
        self.state = State.DRIVE
        self.t_state = 0.0
        self.cooldown_until = -1e9
        self.last_obstacle_t: Optional[float] = None
        self.virtual_lane: Optional[VirtualLane] = None
        self._t_prev: Optional[float] = None
        self._last_out = (0.0, 0.0)

    def _enter(self, s: State, t: float) -> None:
        if s != self.state:
            self.state = s
            self.t_state = t

    def obstacle_ahead(self, est: LaneEstimate) -> bool:
        c = self.cfg
        for _cls, x, y in est.obstacles:
            if 0.0 < x < c.obstacle_trigger_x:
                yc = float(est.center_y(x)) if est.valid else 0.0
                if abs(y - yc) < c.obstacle_lane_margin:
                    return True
        return False

    def _make_virtual_lane(self, est: LaneEstimate) -> VirtualLane:
        x_ref = est.stopline_x if est.stopline_x is not None else 0.0
        return VirtualLane(est.center if est.valid else None, est.stopline_heading, x_ref, self.cfg.cross_actuator_tau)

    def _cross_command(self) -> Tuple[float, float]:
        v = self.cfg.cross_speed
        if self.virtual_lane is None:
            return v, 0.0
        tx, ty = self.virtual_lane.target(self.lookahead)
        ld = max(math.hypot(tx, ty), 1e-3)
        kappa = 2.0 * math.sin(math.atan2(ty, tx)) / ld
        return v, float(np.clip(v * kappa, -3.0, 3.0))

    def update(self, est: LaneEstimate, cmd: Tuple[float, float], t: float, lost: bool) -> Tuple[float, float, State]:
        out = self._update(est, cmd, t, lost)
        # dead reckoning of the virtual lane with the command actually issued last step
        if self.virtual_lane is not None and self._t_prev is not None and self.state == State.CROSS:
            dt = max(0.0, min(0.2, t - self._t_prev))
            self.virtual_lane.move(self._last_out[0], self._last_out[1], dt)
        self._t_prev = t
        self._last_out = (out[0], out[1])
        return out

    def _update(self, est: LaneEstimate, cmd: Tuple[float, float], t: float, lost: bool) -> Tuple[float, float, State]:
        c = self.cfg
        if c.enable_obstacles and self.obstacle_ahead(est):
            self.last_obstacle_t = t
            if self.state != State.STOP_LINE_WAIT:
                self._enter(State.OBSTACLE_STOP, t)

        if self.state == State.OBSTACLE_STOP:
            if self.last_obstacle_t is not None and t - self.last_obstacle_t < c.obstacle_clear_time:
                return 0.0, 0.0, self.state
            self._enter(State.DRIVE, t)

        if self.state == State.STOP_LINE_WAIT:
            if est.valid or est.stopline_heading is not None:  # standing still: keep the freshest measurement
                self.virtual_lane = self._make_virtual_lane(est)
            if t - self.t_state < c.stop_duration:
                return 0.0, 0.0, self.state
            self._enter(State.CROSS, t)

        if self.state == State.CROSS:
            if t - self.t_state < c.cross_duration:
                v, w = self._cross_command()
                return v, w, self.state
            self.cooldown_until = t + c.stopline_cooldown
            self.virtual_lane = None
            self._enter(State.DRIVE, t)

        if (
            c.enable_stoplines
            and est.stopline_x is not None
            and est.stopline_x < c.stopline_trigger_x
            and t >= self.cooldown_until
        ):
            self._enter(State.STOP_LINE_WAIT, t)
            self.virtual_lane = self._make_virtual_lane(est)
            return 0.0, 0.0, self.state

        self._enter(State.LOST if lost else State.DRIVE, t)
        return (0.0, 0.0, self.state) if lost else (cmd[0], cmd[1], self.state)
