"""High-level behaviour: lane following + stop lines + obstacle stops."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from ..config import BehaviorConfig
from ..estimation.lane_estimator import LaneEstimate


class State(str, Enum):
    DRIVE = "DRIVE"
    STOP_LINE_WAIT = "STOP_LINE_WAIT"
    CROSS = "CROSS"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    LOST = "LOST"


class BehaviorFSM:
    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = State.DRIVE
        self.t_state = 0.0
        self.cooldown_until = -1e9
        self.last_obstacle_t: Optional[float] = None

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

    def update(self, est: LaneEstimate, cmd: Tuple[float, float], t: float, lost: bool) -> Tuple[float, float, State]:
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
            if t - self.t_state < c.stop_duration:
                return 0.0, 0.0, self.state
            self._enter(State.CROSS, t)

        if self.state == State.CROSS:
            if t - self.t_state < c.cross_duration:
                return c.cross_speed, 0.0, self.state
            self.cooldown_until = t + c.stopline_cooldown
            self._enter(State.DRIVE, t)

        if (
            c.enable_stoplines
            and est.stopline_x is not None
            and est.stopline_x < c.stopline_trigger_x
            and t >= self.cooldown_until
        ):
            self._enter(State.STOP_LINE_WAIT, t)
            return 0.0, 0.0, self.state

        self._enter(State.LOST if lost else State.DRIVE, t)
        return (0.0, 0.0, self.state) if lost else (cmd[0], cmd[1], self.state)
