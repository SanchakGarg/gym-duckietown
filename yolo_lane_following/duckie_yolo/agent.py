"""The lane-following agent: image in, (v, omega) out.

Identical in simulation and on the robot; only the segmenter weights/backend and the
camera model (rectification + homography) differ.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import AgentConfig
from .control.behavior import BehaviorFSM, State
from .control.pure_pursuit import PurePursuitController
from .estimation.lane_estimator import LaneEstimate, LaneEstimator
from .perception.camera_model import CameraModel
from .perception.segmenters import Segmentation


@dataclass
class AgentStep:
    action: np.ndarray  # [v, omega]
    state: State
    estimate: LaneEstimate
    segmentation: Segmentation
    image: np.ndarray  # rectified RGB image the agent actually saw
    timings: Dict[str, float]


class LaneFollowingAgent:
    def __init__(self, segmenter, camera: CameraModel, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.segmenter = segmenter
        self.camera = camera
        self.estimator = LaneEstimator(camera, self.cfg.estimator, self.cfg.lane)
        self.controller = PurePursuitController(self.cfg.controller)
        self.fsm = BehaviorFSM(self.cfg.behavior)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.fsm.reset()
        self._last_cmd: Tuple[float, float] = (0.0, 0.0)
        self._last_valid_t: Optional[float] = None

    def act(self, rgb: np.ndarray, t: Optional[float] = None, segmentation: Optional[Segmentation] = None) -> AgentStep:
        """rgb: raw camera image (RGB uint8). t: time in seconds (sim time in simulation).

        ``segmentation`` may be supplied to bypass the segmenter (sim oracle / replay).
        """
        t = time.monotonic() if t is None else float(t)
        tm: Dict[str, Any] = {}
        t0 = time.perf_counter()
        img = self.camera.rectify(rgb)
        t1 = time.perf_counter()
        seg = segmentation if segmentation is not None else self.segmenter.segment(img)
        t2 = time.perf_counter()
        est = self.estimator.estimate(seg, lookahead=self.cfg.controller.lookahead)
        t3 = time.perf_counter()

        c = self.cfg.controller
        lost = False
        if est.valid and est.confidence >= c.min_confidence:
            v, w = self.controller(est)
            self._last_cmd = (v, w)
            self._last_valid_t = t
        elif self._last_valid_t is not None and t - self._last_valid_t < c.lost_timeout:
            v, w = self._last_cmd[0] * c.lost_speed_scale, self._last_cmd[1]
        else:
            v, w, lost = 0.0, 0.0, True
        v, w, state = self.fsm.update(est, (v, w), t, lost)
        act = self.cfg.actuation
        v, w = v * act.v_gain, w * act.omega_gain
        t4 = time.perf_counter()
        tm.update(rectify=t1 - t0, segment=t2 - t1, estimate=t3 - t2, control=t4 - t3, total=t4 - t0)
        return AgentStep(np.array([v, w], dtype=float), state, est, seg, img, tm)
