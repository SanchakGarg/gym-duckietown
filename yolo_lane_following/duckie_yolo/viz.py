"""Debug visualisation: mask overlay on the camera image + bird's-eye view of the estimate."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .classes import CLASS_NAMES, OBSTACLE_CLASSES, VIZ_BGR, WHITE, YELLOW, RED
from .estimation.lane_estimator import LaneEstimate
from .perception.camera_model import CameraModel
from .perception.segmenters import Segmentation


def overlay(rgb: np.ndarray, seg: Segmentation, alpha: float = 0.55) -> np.ndarray:
    """BGR image with class masks blended on top."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    color = np.zeros_like(bgr)
    any_m = np.zeros((h, w), bool)
    for c, m in seg.class_masks.items():
        if not m.any():
            continue
        big = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color[big] = VIZ_BGR.get(c, (255, 0, 255))
        any_m |= big
    out = bgr.copy()
    out[any_m] = (alpha * color[any_m] + (1 - alpha) * bgr[any_m]).astype(np.uint8)
    return out


def draw_lane_on_image(img_bgr: np.ndarray, est: LaneEstimate, camera: CameraModel) -> None:
    if est.valid and est.center is not None:
        xs = np.linspace(0.12, 0.7, 30)
        uv, ok = camera.ground_to_image(np.column_stack([xs, est.center_y(xs)]))
        pts = uv[ok].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img_bgr, [pts], False, (0, 255, 0), 2)
        if est.lookahead is not None:
            p, ok = camera.ground_to_image(np.array([est.lookahead]))
            if ok[0]:
                cv2.circle(img_bgr, tuple(int(a) for a in p[0]), 6, (0, 0, 255), -1)


def birdseye(est: LaneEstimate, size=(240, 360), px_per_m: float = 380.0) -> np.ndarray:
    """Top-down view: robot at bottom-center, x forward is up, y left is left."""
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    ox, oy = w // 2, h - 20

    def to_px(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.column_stack([ox - xy[:, 1] * px_per_m, oy - xy[:, 0] * px_per_m]).astype(np.int32)

    for gx in np.arange(0.0, 1.0, 0.1):  # 10 cm grid
        y = int(oy - gx * px_per_m)
        cv2.line(img, (0, y), (w, y), (55, 55, 55), 1)
    for gy in np.arange(-0.6, 0.61, 0.1):
        x = int(ox - gy * px_per_m)
        cv2.line(img, (x, 0), (x, h), (55, 55, 55), 1)
    for c, pts in est.points.items():
        for p in to_px(pts[:: max(1, len(pts) // 300)]) if len(pts) else []:
            cv2.circle(img, (int(p[0]), int(p[1])), 1, VIZ_BGR.get(c, (255, 0, 255)), -1)
    xs = np.linspace(0.0, 0.8, 40)
    for fit, col in ((est.yellow, VIZ_BGR[YELLOW]), (est.white, VIZ_BGR[WHITE])):
        if fit is not None:
            cv2.polylines(img, [to_px(np.column_stack([xs, fit.y(xs)]))], False, col, 1)
    if est.valid and est.center is not None:
        cv2.polylines(img, [to_px(np.column_stack([xs, est.center_y(xs)]))], False, (0, 255, 0), 2)
        if est.lookahead is not None:
            cv2.circle(img, tuple(to_px(est.lookahead)[0]), 5, (0, 0, 255), -1)
    if est.stopline_x is not None:
        y = int(oy - est.stopline_x * px_per_m)
        cv2.line(img, (0, y), (w, y), VIZ_BGR[RED], 2)
    for cls, x, y in est.obstacles:
        cv2.circle(img, tuple(to_px((x, y))[0]), 7, VIZ_BGR.get(cls, (255, 0, 255)), 2)
    # robot
    cv2.fillPoly(img, [np.array([[ox, oy - 14], [ox - 8, oy + 6], [ox + 8, oy + 6]], np.int32)], (200, 200, 200))
    return img


def compose(
    rgb: np.ndarray,
    seg: Segmentation,
    est: LaneEstimate,
    camera: Optional[CameraModel] = None,
    text_lines=(),
) -> np.ndarray:
    """Side-by-side debug frame (BGR)."""
    left = overlay(rgb, seg)
    if camera is not None:
        draw_lane_on_image(left, est, camera)
    h = left.shape[0]
    bev = birdseye(est, size=(int(h * 0.66), h))
    out = np.hstack([left, bev])
    y = 22
    for line in text_lines:
        cv2.putText(out, str(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def legend() -> str:
    return ", ".join(f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)) + f" (obstacles: {OBSTACLE_CLASSES})"
