"""Sanity check: do auto-generated labels align with the RGB images?

Only meaningful for datasets generated with --no-domain-rand (plain colours), where a simple
HSV threshold of the RGB image is a reliable reference for yellow/white/red markings.

    python tools/check_label_alignment.py data/align
"""
import glob
import os
import sys

import cv2
import numpy as np

from duckie_yolo.sim.labels import yolo_lines_to_label_map

REF = {
    0: lambda hsv: cv2.inRange(hsv, (15, 80, 80), (40, 255, 255)) > 0,  # yellow
    1: lambda hsv: cv2.inRange(hsv, (0, 0, 170), (180, 50, 255)) > 0,  # white
    2: lambda hsv: (cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (172, 120, 80), (180, 255, 255))) > 0,
}


def main(root):
    ious = {c: [] for c in REF}
    for img_path in sorted(glob.glob(os.path.join(root, "images", "*", "*.jpg"))):
        lab_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        lines = [l for l in open(lab_path).read().splitlines() if l.strip()]
        lm = yolo_lines_to_label_map(lines, w, h)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        horizon = int(0.3 * h)
        for c, ref in REF.items():
            r = ref(hsv)
            l = lm == c
            r[:horizon] = False
            l[:horizon] = False
            if r.sum() < 200 and l.sum() < 200:
                continue
            ious[c].append((r & l).sum() / max(1, (r | l).sum()))
    for c, v in ious.items():
        if v:
            print(f"class {c}: mean IoU {np.mean(v):.3f}  median {np.median(v):.3f}  n={len(v)}")


if __name__ == "__main__":
    main(sys.argv[1])
