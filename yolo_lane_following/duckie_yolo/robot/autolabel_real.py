"""Pre-label real (rectified) images with the sim-trained model, for correction + fine-tuning.

    python -m duckie_yolo.robot.autolabel_real --images data/real/raw --weights runs/segment/sim_v1/weights/best.pt --out data/real/labeled

Writes a YOLO-seg dataset (images/ + labels/ + previews/). Import it into CVAT or Label Studio
(YOLO segmentation format), fix the masks, export again, then fine-tune on sim + real:

    python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --extra-train data/real/labeled/images \
        --model runs/segment/sim_v1/weights/best.pt --epochs 30 --name sim_real_v1

``--segmenter color`` uses HSV thresholds instead (useful before any model exists).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from ..classes import CLASS_NAMES
from ..config import SegmenterConfig
from ..perception.segmenters import ColorSegmenter, YoloSegmenter
from ..sim.generate_dataset import preview, write_dataset_yaml
from ..sim.labels import label_map_to_polygons, polygons_to_yolo_lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--segmenter", default="yolo", choices=["yolo", "color"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=320)
    a = ap.parse_args(argv)

    files = sorted(sum((glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.png")), []))
    for d in ("images/train", "labels/train", "previews"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    if a.segmenter == "yolo":
        seg = YoloSegmenter(SegmenterConfig(weights=a.weights, imgsz=a.imgsz, conf=a.conf), work_scale=1.0)
    else:
        seg = ColorSegmenter(work_scale=1.0)
    for i, f in enumerate(files):
        bgr = cv2.imread(f)
        h, w = bgr.shape[:2]
        s = seg.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        lm = np.full((h, w), -1, np.int16)
        for c, m in s.class_masks.items():
            lm[cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)] = c
        lines = polygons_to_yolo_lines(label_map_to_polygons(lm, min_area=15), w, h)
        name = os.path.splitext(os.path.basename(f))[0]
        shutil.copy(f, os.path.join(a.out, "images/train", os.path.basename(f)))
        with open(os.path.join(a.out, "labels/train", name + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        if i % 10 == 0:
            cv2.imwrite(os.path.join(a.out, "previews", name + ".jpg"), preview(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), lines))
    write_dataset_yaml(a.out)
    print(f"[autolabel] labelled {len(files)} images -> {a.out} (classes: {CLASS_NAMES})")


if __name__ == "__main__":
    main()
