"""Pure-pursuit steering towards a lookahead point on the estimated lane center."""
from __future__ import annotations

import math
from typing import Tuple

from ..config import ControllerConfig
from ..estimation.lane_estimator import LaneEstimate


class PurePursuitController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg

    def __call__(self, est: LaneEstimate) -> Tuple[float, float]:
        """Returns (v, omega): forward speed and yaw rate commands (omega > 0 turns left)."""
        c = self.cfg
        if not est.valid or est.lookahead is None:
            return 0.0, 0.0
        tx, ty = est.lookahead
        ld = max(math.hypot(tx, ty), 1e-3)
        alpha = math.atan2(ty, tx)
        kappa = 2.0 * math.sin(alpha) / ld
        v = c.v_ref / (1.0 + c.curvature_gain * abs(kappa))
        v *= 0.6 + 0.4 * max(0.0, min(1.0, est.confidence))
        v = max(v, c.v_min)
        omega = max(-c.omega_max, min(c.omega_max, v * kappa))
        return v, omega
