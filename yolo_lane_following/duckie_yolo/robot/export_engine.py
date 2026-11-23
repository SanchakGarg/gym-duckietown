"""Build a TensorRT engine ON THE JETSON (engines are specific to the GPU + TensorRT version).

    python3 -m duckie_yolo.robot.export_engine --weights models/best.pt --imgsz 320
    # -> models/best.engine ; then: run_robot --weights models/best.engine

Run it inside the Ultralytics JetPack container (see robot/docker/Dockerfile.agent).
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--fp32", action="store_true", help="disable FP16")
    ap.add_argument("--workspace", type=float, default=1.0, help="TensorRT workspace (GB); keep small on a 4 GB Orin Nano")
    ap.add_argument("--benchmark", type=int, default=100, help="timed inferences after export (0 = skip)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    path = YOLO(a.weights).export(format="engine", imgsz=a.imgsz, half=not a.fp32, device=0, workspace=a.workspace, simplify=True)
    print(f"[export] engine: {path}")
    if a.benchmark:
        m = YOLO(path, task="segment")
        img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        for _ in range(10):
            m.predict(img, imgsz=a.imgsz, verbose=False)
        t0 = time.perf_counter()
        for _ in range(a.benchmark):
            m.predict(img, imgsz=a.imgsz, verbose=False)
        dt = (time.perf_counter() - t0) / a.benchmark
        print(f"[export] {1000 * dt:.1f} ms / image ({1 / dt:.0f} FPS) incl. pre/post-processing")


if __name__ == "__main__":
    main()
