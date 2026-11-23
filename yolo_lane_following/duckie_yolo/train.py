"""Train the YOLO-seg lane-marking model (Ultralytics) and export it.

    python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --epochs 60 --name sim_v1
    # fine-tune with corrected real images (see robot/autolabel_real.py):
    python -m duckie_yolo.train --data data/sim_v1/dataset.yaml --extra-train data/real/labeled/images \
        --model runs/segment/sim_v1/weights/best.pt --epochs 30 --name sim_real_v1

Augmentation favours sim-to-real transfer: strong colour/brightness jitter on top of the
simulator's own domain randomization. ONNX export is portable; TensorRT engines must be built
on the Jetson itself (robot/export_engine.py).
"""
from __future__ import annotations

import argparse
import os

import yaml


def _merged_data_yaml(data: str, extra_train, out_dir: str) -> str:
    if not extra_train:
        return data
    with open(data) as f:
        d = yaml.safe_load(f)
    root = d.get("path", os.path.dirname(os.path.abspath(data)))
    train = d["train"] if isinstance(d["train"], list) else [d["train"]]
    train = [t if os.path.isabs(t) else os.path.join(root, t) for t in train]
    train += [os.path.abspath(p) for p in extra_train]
    d["train"] = train
    for k in ("val", "test"):
        if k in d and not os.path.isabs(d[k]):
            d[k] = os.path.join(root, d[k])
    d.pop("path", None)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "merged_dataset.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return path


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra-train", nargs="*", default=[], help="extra image folders (labels/ next to images/)")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained checkpoint or a previous best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--project", default="runs/segment")
    ap.add_argument("--name", default="sim_v1")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--close-mosaic", type=int, default=10, help="last N epochs without mosaic augmentation")
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set (smoke tests)")
    ap.add_argument("--export", nargs="*", default=["onnx"], help="formats to export after training")
    ap.add_argument("--cache", default="ram", help="ram | disk | none (image cache)")
    ap.add_argument("--no-final-val", action="store_true")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    data = _merged_data_yaml(a.data, a.extra_train, os.path.join(a.project, a.name + "_data"))
    model = YOLO(a.model)
    model.train(
        data=data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        workers=a.workers,
        device=a.device or default_device(),
        project=os.path.abspath(a.project),  # ultralytics>=8.4 nests relative paths under its runs dir
        name=a.name,
        exist_ok=True,
        patience=a.patience,
        fraction=a.fraction,
        # augmentation: colour/brightness jitter for sim2real, mild geometry
        hsv_h=0.04,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0003,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=a.close_mosaic,
        plots=True,
        cache=False if a.cache == 'none' else a.cache,
    )
    best = str(model.trainer.best)
    print(f"[train] best weights: {best}")
    if a.no_final_val:
        a.export = a.export or []
    metrics = None if a.no_final_val else YOLO(best).val(data=data, imgsz=a.imgsz, split="test" if "test" in yaml.safe_load(open(data)) else "val",
                             device=a.device or default_device(), plots=False)
    try:
        if metrics is None:
            raise RuntimeError("skipped")
        print(f"[train] test mask mAP50={metrics.seg.map50:.3f} mAP50-95={metrics.seg.map:.3f}")
        for i, n in metrics.names.items():
            print(f"    {n:12s} mask mAP50={metrics.seg.ap50[i]:.3f}" if i < len(metrics.seg.ap50) else f"    {n}: n/a")
    except Exception as e:  # metrics API differences across ultralytics versions
        print("[train] metrics:", e)
    for fmt in a.export:
        path = YOLO(best).export(format=fmt, imgsz=a.imgsz, simplify=True) if fmt == "onnx" else YOLO(best).export(format=fmt, imgsz=a.imgsz)
        print(f"[train] exported {fmt}: {path}")


if __name__ == "__main__":
    main()
