"""Lane pose estimation from segmented lane markings.

Pipeline: class masks -> sampled pixels -> ground points (homography) -> robust polynomial
fits of the yellow and white lines -> offset to the lane center -> fused center curve
y_c(x) in the robot frame (x forward, y left).

Sign conventions match ``gym_duckietown.Simulator.get_lane_pos2``:
    d   > 0  : robot is to the RIGHT of the lane center (center line is on its left)
    phi > 0  : robot heading points to the RIGHT of the lane direction
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..classes import OBSTACLE_CLASSES, RED, WHITE, YELLOW
from ..config import EstimatorConfig, LaneGeometry
from ..perception.camera_model import CameraModel
from ..perception.segmenters import Segmentation


@dataclass
class PolyFit:
    coeffs: np.ndarray  # np.polyval order (highest power first)
    x_lo: float
    x_hi: float
    n_inliers: int

    def y(self, x):
        return np.polyval(self.coeffs, x)

    def dy(self, x):
        return np.polyval(np.polyder(self.coeffs), x)


@dataclass
class LaneEstimate:
    valid: bool = False
    d: float = 0.0
    phi: float = 0.0
    center: Optional[np.ndarray] = None  # poly coeffs of the lane center y_c(x)
    lookahead: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    source: str = "none"  # both | yellow | white | none
    yellow: Optional[PolyFit] = None
    white: Optional[PolyFit] = None
    stopline_x: Optional[float] = None
    # direction of the lane beyond the stop line (rad, robot frame, + = left): the stop line is
    # perpendicular to the lane, so its normal gives the crossing direction
    stopline_heading: Optional[float] = None
    obstacles: List[Tuple[int, float, float]] = field(default_factory=list)  # (cls, x, y)
    points: Dict[int, np.ndarray] = field(default_factory=dict)  # ground points per class (viz)

    def center_y(self, x):
        if self.center is None:
            return np.zeros_like(np.asarray(x, dtype=float))
        return np.polyval(self.center, x)


def ransac_polyfit(
    x: np.ndarray, y: np.ndarray, deg: int, iters: int, thresh: float, rng: np.random.Generator
) -> Optional[PolyFit]:
    n = len(x)
    if n < deg + 2:
        return None
    best_inl = None
    best_cnt = -1
    for _ in range(iters):
        idx = rng.choice(n, deg + 1, replace=False)
        xs = x[idx]
        if np.ptp(xs) < 1e-3:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # RankWarning for nearly-degenerate minimal samples
                c = np.polyfit(xs, y[idx], deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        inl = np.abs(np.polyval(c, x) - y) < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None or best_cnt < deg + 2:
        return None
    xi, yi = x[best_inl], y[best_inl]
    c = np.polyfit(xi, yi, deg)
    # one refinement pass with the final model
    inl = np.abs(np.polyval(c, x) - y) < thresh
    if inl.sum() >= deg + 2:
        xi, yi = x[inl], y[inl]
        c = np.polyfit(xi, yi, deg)
    return PolyFit(c, float(xi.min()), float(xi.max()), int(len(xi)))


def _pad_poly(c: np.ndarray, deg: int = 2) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if len(c) < deg + 1:
        c = np.concatenate([np.zeros(deg + 1 - len(c)), c])
    return c


class LaneEstimator:
    def __init__(self, camera: CameraModel, cfg: EstimatorConfig, lane: LaneGeometry, seed: int = 0):
        self.camera = camera
        self.cfg = cfg
        self.lane = lane
        self.rng = np.random.default_rng(seed)
        self._horizon = camera.horizon_row()
        self.reset()

    def reset(self) -> None:
        self._ema: Optional[np.ndarray] = None
        self._misses = 0

    # ------------------------------------------------------------------ points
    def mask_pixels(self, mask: np.ndarray, scale: float, max_points: Optional[int] = None) -> np.ndarray:
        """Sub-sampled pixel coordinates (full image resolution, u/v) of a work-resolution mask."""
        st = max(1, int(self.cfg.stride))
        # Rows where the marking runs off the left/right image border only contain part of the
        # line's width, which biases the fit towards the image centre: ignore them.
        edge_rows = mask[:, :2].any(axis=1) | mask[:, -2:].any(axis=1)
        if edge_rows.any():
            mask = mask.copy()
            mask[edge_rows] = False
        ys, xs = np.nonzero(mask[::st, ::st])
        if len(xs) == 0:
            return np.zeros((0, 2))
        u = (xs * st + 0.5) / scale
        v = (ys * st + 0.5) / scale
        keep = v > self._horizon + 2.0
        uv = np.column_stack([u[keep], v[keep]])
        mp = max_points or self.cfg.max_points
        if len(uv) > mp:
            uv = uv[np.linspace(0, len(uv) - 1, mp).astype(int)]
        return uv

    def ground_points(self, seg: Segmentation, cls: int) -> np.ndarray:
        m = seg.class_masks.get(cls)
        if m is None or not m.any():
            return np.zeros((0, 2))
        uv = self.mask_pixels(m, seg.work_scale)
        if len(uv) == 0:
            return np.zeros((0, 2))
        pts, valid = self.camera.image_to_ground(uv)
        pts = pts[valid]
        c = self.cfg
        roi = (pts[:, 0] >= c.x_min) & (pts[:, 0] <= c.x_max) & (np.abs(pts[:, 1]) <= c.y_abs_max)
        return pts[roi]

    # ------------------------------------------------------------------ fitting
    def _fit(self, pts: np.ndarray) -> Optional[PolyFit]:
        c = self.cfg
        if len(pts) < c.min_points:
            return None
        deg = c.poly_degree if np.ptp(pts[:, 0]) >= c.min_x_span_quadratic else 1
        fit = ransac_polyfit(pts[:, 0], pts[:, 1], deg, c.ransac_iters, c.ransac_thresh, self.rng)
        if fit is None or deg < 2:
            return fit
        # Occam: keep the quadratic only if it explains the inliers clearly better than a line
        inl = np.abs(fit.y(pts[:, 0]) - pts[:, 1]) < c.ransac_thresh
        x, y = pts[inl, 0], pts[inl, 1]
        lin = np.polyfit(x, y, 1)
        rms1 = float(np.sqrt(np.mean((np.polyval(lin, x) - y) ** 2)))
        rms2 = float(np.sqrt(np.mean((fit.y(x) - y) ** 2)))
        if rms1 < c.quadratic_gain * rms2 + 0.002:
            return PolyFit(lin, fit.x_lo, fit.x_hi, fit.n_inliers)
        return fit

    def _offset_points(self, fit: PolyFit, signed_offset: float) -> np.ndarray:
        """Points of the curve shifted by ``signed_offset`` along its left normal."""
        c = self.cfg
        lo = max(c.x_min, fit.x_lo - 0.05)
        hi = min(c.x_max, fit.x_hi + 0.10)
        if hi - lo < 0.02:
            hi = lo + 0.02
        x = np.linspace(lo, hi, 12)
        y = fit.y(x)
        s = fit.dy(x)
        nrm = np.sqrt(1.0 + s * s)
        nx, ny = -s / nrm, 1.0 / nrm
        return np.column_stack([x + signed_offset * nx, y + signed_offset * ny])

    @staticmethod
    def _fit_center(pts: np.ndarray, w: np.ndarray, deg: int) -> np.ndarray:
        sw = np.sqrt(w)
        lin = np.polyfit(pts[:, 0], pts[:, 1], 1, w=sw)
        if deg < 2 or np.ptp(pts[:, 0]) < 0.15:
            return lin
        quad = np.polyfit(pts[:, 0], pts[:, 1], 2, w=sw)
        r1 = np.sqrt(np.average((np.polyval(lin, pts[:, 0]) - pts[:, 1]) ** 2, weights=w))
        r2 = np.sqrt(np.average((np.polyval(quad, pts[:, 0]) - pts[:, 1]) ** 2, weights=w))
        return lin if r1 < 1.5 * r2 + 0.002 else quad

    # ------------------------------------------------------------------ main
    def estimate(self, seg: Segmentation, lookahead: float = 0.3) -> LaneEstimate:
        c = self.cfg
        est = LaneEstimate()
        p_y = self.ground_points(seg, YELLOW)
        p_w = self.ground_points(seg, WHITE)
        est.points = {YELLOW: p_y, WHITE: p_w}

        fy = self._fit(p_y)
        if fy is not None and len(p_w):
            # the white line we want is to the right of the yellow one; points on the left
            # belong to the far edge of the opposite lane
            p_w = p_w[p_w[:, 1] < fy.y(p_w[:, 0]) - c.white_min_right_of_yellow]
            est.points[WHITE] = p_w
        fw = self._fit(p_w)
        est.yellow, est.white = fy, fw

        cands = []
        if fy is not None:
            cands.append(("yellow", self._offset_points(fy, -self.lane.yellow_offset), fy.n_inliers))
        if fw is not None:
            white_off = +self.lane.white_offset
            if fy is None and self._ema is not None:
                # A lone white line is ambiguous: our right edge, or the left edge of the road (seen
                # when turned left / drifting into the opposite lane). Pick the hypothesis that is
                # consistent with the lane centre we were tracking. (A lone yellow line is not
                # ambiguous: our lane is always to its right.)
                left_edge_off = -(self.lane.white_offset + 2.0 * self.lane.yellow_offset)
                x_ref = float(np.clip(0.5 * (fw.x_lo + fw.x_hi), c.x_min, c.x_max))
                prev = float(np.polyval(self._ema, x_ref))
                y_w = float(fw.y(x_ref))
                if abs(y_w + left_edge_off - prev) < abs(y_w + white_off - prev):
                    white_off = left_edge_off
                    est.source = "white(left edge)"
            cands.append(("white", self._offset_points(fw, white_off), fw.n_inliers))

        if len(cands) == 2:
            cy = self._fit_center(cands[0][1], np.ones(len(cands[0][1])), 1)
            cw = self._fit_center(cands[1][1], np.ones(len(cands[1][1])), 1)
            if abs(np.polyval(cy, 0.3) - np.polyval(cw, 0.3)) > c.max_center_disagreement:
                # trust the yellow line unless it is clearly weaker
                keep = 0 if cands[0][2] >= 0.5 * cands[1][2] else 1
                cands = [cands[keep]]

        meas = None
        if cands:
            pts = np.vstack([p for _, p, _ in cands])
            wts = np.concatenate([np.full(len(p), float(n)) for _, p, n in cands])
            meas = _pad_poly(self._fit_center(pts, wts, c.poly_degree))
            if est.source != "white(left edge)":
                est.source = "both" if len(cands) == 2 else cands[0][0]

        if meas is not None:
            self._misses = 0
            if self._ema is None:
                self._ema = meas
            else:
                a = c.ema_alpha
                self._ema = a * meas + (1 - a) * self._ema
            n_inl = sum(n for _, _, n in cands)
            est.confidence = float(min(1.0, n_inl / 80.0 + (0.2 if est.source == "both" else 0.0)))
        else:
            self._misses += 1
            if self._misses > 15:
                self._ema = None

        if meas is not None and self._ema is not None:
            est.valid = True
            est.center = self._ema.copy()
            y0 = float(np.polyval(est.center, 0.0))
            slope = float(np.polyval(np.polyder(est.center), 0.0))
            est.phi = math.atan(slope)
            est.d = y0 * math.cos(est.phi)
            est.lookahead = self.lookahead_point(est.center, lookahead)

        self._parse_stopline(seg, est)
        self._parse_obstacles(seg, est)
        return est

    @staticmethod
    def lookahead_point(center: np.ndarray, L: float) -> Tuple[float, float]:
        xs = np.linspace(0.0, 1.5 * L, 300)
        ys = np.polyval(center, xs)
        i = int(np.argmin(np.abs(np.hypot(xs, ys) - L)))
        return float(xs[i]), float(ys[i])

    # ------------------------------------------------------------------ stop line / obstacles
    def _parse_stopline(self, seg: Segmentation, est: LaneEstimate) -> None:
        p_r = self.ground_points(seg, RED)
        est.points[RED] = p_r
        if len(p_r) < self.cfg.min_red_points:
            return
        yc = est.center_y(p_r[:, 0]) if est.valid else np.zeros(len(p_r))
        p = p_r[np.abs(p_r[:, 1] - yc) < self.cfg.stopline_corridor]
        if len(p) >= self.cfg.min_red_points:
            est.stopline_x = float(np.percentile(p[:, 0], 10))
            # orientation of the nearest red line: fit x = a + b*y (the line runs across the lane)
            near = p[p[:, 0] < est.stopline_x + 0.08]
            if len(near) >= self.cfg.min_red_points and np.ptp(near[:, 1]) > 0.08:
                fit = ransac_polyfit(near[:, 1], near[:, 0], 1, self.cfg.ransac_iters, 0.015, self.rng)
                if fit is not None:
                    b = float(fit.coeffs[0])
                    est.stopline_heading = -math.atan(b)

    def _parse_obstacles(self, seg: Segmentation, est: LaneEstimate) -> None:
        s = seg.work_scale
        for inst in seg.instances:
            if inst.cls not in OBSTACLE_CLASSES or inst.mask.sum() < self.cfg.obstacle_min_area_px:
                continue
            ys, xs = np.nonzero(inst.mask)
            vb = ys.max()
            u = float(np.median(xs[ys >= vb - 2]))
            pt, valid = self.camera.image_to_ground(np.array([[(u + 0.5) / s, (vb + 1.0) / s]]))
            if valid[0]:
                est.obstacles.append((int(inst.cls), float(pt[0, 0]), float(pt[0, 1])))
