"""Segmenters turn an RGB image into per-class binary masks (+ obstacle instances).

* ``YoloSegmenter``  - Ultralytics YOLO-seg model (.pt / .onnx / TensorRT .engine)
* ``ColorSegmenter`` - classical HSV thresholds (baseline, and a bootstrap auto-labeller for real images)
* ``segmentation_from_label_map`` - wraps a ground-truth label map (sim "oracle" / tests)

Masks are produced at ``work_scale`` x the input resolution to keep the geometry cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..classes import DUCKIE, DUCKIEBOT, NUM_CLASSES, OBSTACLE_CLASSES, RED, WHITE, YELLOW
from ..config import SegmenterConfig


@dataclass
class Instance:
    cls: int
    conf: float
    mask: np.ndarray  # bool, work resolution


@dataclass
class Segmentation:
    image_size: Tuple[int, int]  # (width, height) of the input (rectified) image
    work_scale: float
    class_masks: Dict[int, np.ndarray] = field(default_factory=dict)  # bool, work resolution
    instances: List[Instance] = field(default_factory=list)

    @property
    def work_size(self) -> Tuple[int, int]:
        return work_size(self.image_size, self.work_scale)

    def mask(self, cls: int) -> np.ndarray:
        m = self.class_masks.get(cls)
        if m is None:
            w, h = self.work_size
            return np.zeros((h, w), bool)
        return m


def work_size(image_size: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return max(1, int(round(image_size[0] * scale))), max(1, int(round(image_size[1] * scale)))


def _instances_from_mask(mask: np.ndarray, cls: int, min_area: int) -> List[Instance]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append(Instance(cls, 1.0, labels == i))
    return out


def segmentation_from_label_map(
    label_map: np.ndarray, work_scale: float = 0.5, min_instance_area: int = 20
) -> Segmentation:
    """label_map: (H, W) int array with class ids, -1 (or any id outside the classes) = background."""
    h, w = label_map.shape[:2]
    ww, wh = work_size((w, h), work_scale)
    small = cv2.resize(label_map.astype(np.int16), (ww, wh), interpolation=cv2.INTER_NEAREST)
    seg = Segmentation((w, h), work_scale)
    for c in range(NUM_CLASSES):
        m = small == c
        if m.any():
            seg.class_masks[c] = m
            if c in OBSTACLE_CLASSES:
                seg.instances.extend(_instances_from_mask(m, c, min_instance_area))
    return seg


class YoloSegmenter:
    """Ultralytics YOLO segmentation model wrapper.

    Accepts RGB uint8 images (the convention used everywhere in duckie_yolo / gym) and
    converts to BGR for Ultralytics, which treats numpy inputs as OpenCV BGR images.
    """

    def __init__(self, cfg: SegmenterConfig, work_scale: float = 0.5):
        from ultralytics import YOLO

        if not cfg.weights:
            raise ValueError("SegmenterConfig.weights must point to a .pt/.onnx/.engine file")
        self.cfg = cfg
        self.work_scale = work_scale
        self.model = YOLO(cfg.weights, task="segment")
        # FP16 is baked into TensorRT engines at export time (robot/export_engine.py); passing
        # `half` at predict time is deprecated in recent Ultralytics and warns on every frame.
        self._kw = dict(imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False)
        if cfg.device:
            self._kw["device"] = cfg.device
        self.names = getattr(self.model, "names", None)

    def predict_raw(self, rgb: np.ndarray):
        return self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), **self._kw)[0]

    def segment(self, rgb: np.ndarray) -> Segmentation:
        h, w = rgb.shape[:2]
        res = self.predict_raw(rgb)
        return self.from_result(res, (w, h))

    def from_result(self, res, image_size: Tuple[int, int]) -> Segmentation:
        s = self.work_scale
        seg = Segmentation(image_size, s)
        ww, wh = seg.work_size
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return seg
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for poly, c, cf in zip(res.masks.xy, classes, confs):
            if poly is None or len(poly) < 3 or c >= NUM_CLASSES:
                continue
            m = np.zeros((wh, ww), np.uint8)
            cv2.fillPoly(m, [np.round(np.asarray(poly) * s).astype(np.int32)], 1)
            mb = m.astype(bool)
            if c in seg.class_masks:
                seg.class_masks[c] |= mb
            else:
                seg.class_masks[c] = mb
            if c in OBSTACLE_CLASSES:
                seg.instances.append(Instance(int(c), float(cf), mb))
        return seg


@dataclass
class HSVRange:
    lo: Sequence[int]
    hi: Sequence[int]


class ColorSegmenter:
    """Classical HSV colour thresholding of lane markings (no obstacles).

    Useful as a baseline, for debugging the geometry on the real robot before a model is
    trained, and to bootstrap labels for real images (see robot/autolabel_real.py).
    Ranges are OpenCV HSV (H in [0,180)). Tune them for your lighting.
    """

    def __init__(
        self,
        work_scale: float = 0.5,
        yellow: HSVRange = HSVRange((15, 80, 100), (40, 255, 255)),
        white: HSVRange = HSVRange((0, 0, 160), (180, 60, 255)),
        red1: HSVRange = HSVRange((0, 110, 90), (8, 255, 255)),
        red2: HSVRange = HSVRange((170, 110, 90), (180, 255, 255)),
        horizon_row_frac: float = 0.25,
    ):
        self.work_scale = work_scale
        self.yellow, self.white, self.red1, self.red2 = yellow, white, red1, red2
        self.horizon_row_frac = horizon_row_frac
        self._kernel = np.ones((3, 3), np.uint8)

    def _rng(self, hsv, r: HSVRange) -> np.ndarray:
        return cv2.inRange(hsv, np.array(r.lo, np.uint8), np.array(r.hi, np.uint8))

    def segment(self, rgb: np.ndarray) -> Segmentation:
        h, w = rgb.shape[:2]
        seg = Segmentation((w, h), self.work_scale)
        ww, wh = seg.work_size
        small = cv2.resize(rgb, (ww, wh), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        masks = {
            YELLOW: self._rng(hsv, self.yellow),
            WHITE: self._rng(hsv, self.white),
            RED: self._rng(hsv, self.red1) | self._rng(hsv, self.red2),
        }
        cut = int(self.horizon_row_frac * wh)
        for c, m in masks.items():
            m[:cut] = 0
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
            if m.any():
                seg.class_masks[c] = m.astype(bool)
        return seg


def build_segmenter(cfg: SegmenterConfig, work_scale: float, kind: Optional[str] = None):
    kind = kind or ("yolo" if cfg.weights else "color")
    if kind == "yolo":
        return YoloSegmenter(cfg, work_scale)
    if kind == "color":
        return ColorSegmenter(work_scale)
    raise ValueError(f"unknown segmenter kind {kind!r}")


__all__ = [
    "Instance",
    "Segmentation",
    "segmentation_from_label_map",
    "YoloSegmenter",
    "ColorSegmenter",
    "HSVRange",
    "build_segmenter",
    "DUCKIE",
    "DUCKIEBOT",
]
