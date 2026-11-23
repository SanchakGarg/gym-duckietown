"""Generate an auto-labelled YOLO-seg dataset from gym-duckietown.

A privileged pure-pursuit expert drives (with injected steering noise and deliberate
excursions so that off-center / misaligned views are covered) while every ``--every``-th
frame is saved together with polygon labels extracted from the simulator's
segmentation render. Domain + camera randomization are on by default.

Example:
    python -m duckie_yolo.sim.generate_dataset --out data/sim_v1 --frames 20000
    python -m duckie_yolo.sim.generate_dataset --out data/smoke --frames 60 --maps small_loop --preview-every 10
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import cv2
import numpy as np

from ..classes import CLASS_NAMES
from .expert import PurePursuitExpert
from .labels import (
    clean_label_map,
    env_mesh_names,
    label_map_to_polygons,
    object_palette,
    polygons_to_yolo_lines,
    render_label_map,
    yolo_lines_to_label_map,
)
from .sim_utils import HELDOUT_MAPS, TRAIN_MAPS, gt_lane_pose, make_env

PALETTE_BGR = np.array(
    [[0, 215, 255], [255, 255, 255], [40, 40, 230], [0, 140, 255], [255, 120, 40]], np.uint8
)


def preview(obs_rgb: np.ndarray, lines: List[str]) -> np.ndarray:
    h, w = obs_rgb.shape[:2]
    lm = yolo_lines_to_label_map(lines, w, h)
    bgr = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)
    ov = bgr.copy()
    m = lm >= 0
    ov[m] = (0.4 * bgr[m] + 0.6 * PALETTE_BGR[lm[m]]).astype(np.uint8)
    return np.hstack([bgr, ov])


def write_dataset_yaml(out: str) -> str:
    path = os.path.join(out, "dataset.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n")
        for i, n in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {n}\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=20000, help="total frames for train+val")
    ap.add_argument("--test-frames", type=int, default=-1, help="frames from held-out maps (default 5%%)")
    ap.add_argument("--maps", nargs="*", default=None, help="training maps (default: sim_utils.TRAIN_MAPS)")
    ap.add_argument("--test-maps", nargs="*", default=None)
    ap.add_argument("--every", type=int, default=2, help="save every N-th step")
    ap.add_argument("--episode-steps", type=int, default=400)
    ap.add_argument("--val-every", type=int, default=10, help="every N-th episode goes to val")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-domain-rand", action="store_true")
    ap.add_argument("--no-camera-rand", action="store_true")
    ap.add_argument("--noise", type=float, default=0.3, help="AR(1) steering noise sigma")
    ap.add_argument("--excursion-prob", type=float, default=0.01)
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--preview-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name-prefix", default="", help="prefix for file names (to merge several runs)")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    maps = a.maps or TRAIN_MAPS
    test_maps = a.test_maps if a.test_maps is not None else HELDOUT_MAPS
    n_test = a.test_frames if a.test_frames >= 0 else int(0.05 * a.frames)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(a.out, "previews"), exist_ok=True)
    write_dataset_yaml(a.out)

    meta_name = f"meta_{a.name_prefix.strip('_')}.csv" if a.name_prefix else "meta.csv"
    meta_f = open(os.path.join(a.out, meta_name), "w", newline="")
    meta = csv.writer(meta_f)
    meta.writerow(["split", "name", "map", "episode", "step", "style", "gt_d", "gt_phi", "n_instances"] +
                  [f"n_{c}" for c in CLASS_NAMES])

    envs: Dict[str, object] = {}
    palettes: Dict[str, dict] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    jobs = [("trainval", maps, a.frames), ("test", test_maps, n_test)]
    t_start = time.time()
    ep = 0
    for kind, map_list, target in jobs:
        if target <= 0 or not map_list:
            continue
        saved = 0
        while saved < target:
            m = map_list[ep % len(map_list)]
            split = "test" if kind == "test" else ("val" if ep % a.val_every == 0 else "train")
            if m not in envs:
                envs[m] = make_env(
                    m,
                    seed=int(rng.integers(1 << 30)),
                    domain_rand=not a.no_domain_rand,
                    camera_rand=not a.no_camera_rand,
                    width=a.width,
                    height=a.height,
                    max_steps=10 ** 6,
                    accept_start_angle_deg=50,
                )
                palettes[m] = object_palette(env_mesh_names(envs[m]))
            env = envs[m]
            u = env.unwrapped
            u.style = str(rng.choice(["photos", "synthetic"], p=[0.7, 0.3]))
            u.seed(int(rng.integers(1 << 30)))
            obs = env.reset()
            expert = PurePursuitExpert(env, ref_velocity=float(rng.uniform(0.3, 0.6)))
            noise, exc_left, exc_bias = 0.0, 0, 0.0
            for step in range(a.episode_steps):
                v, w = expert.predict()
                noise = 0.85 * noise + a.noise * float(rng.normal())
                if exc_left == 0 and rng.random() < a.excursion_prob:
                    exc_left = int(rng.integers(6, 16))
                    exc_bias = float(rng.choice([-1, 1]) * rng.uniform(1.5, 3.0))
                if exc_left > 0:
                    exc_left -= 1
                    w = exc_bias
                obs, _r, done, _info = env.step(np.array([v, w + noise]))
                if done:
                    break
                if step < 3 or step % a.every:
                    continue
                img = np.ascontiguousarray(obs).copy()  # obs buffer is reused by the next render
                lm = clean_label_map(render_label_map(env, palettes[m]))
                polys = label_map_to_polygons(lm, min_area=a.min_area)
                lines = polygons_to_yolo_lines(polys, a.width, a.height)
                name = f"{a.name_prefix}{m}_e{ep:05d}_s{step:04d}"
                cv2.imwrite(
                    os.path.join(a.out, "images", split, name + ".jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality],
                )
                with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                gt = gt_lane_pose(env)
                per_cls = [sum(1 for c, _ in polys if c == k) for k in range(len(CLASS_NAMES))]
                meta.writerow([split, name, m, ep, step, u.style,
                               "" if gt is None else f"{gt[0]:.4f}", "" if gt is None else f"{gt[1]:.4f}",
                               len(polys)] + per_cls)
                counts[split] += 1
                saved += 1
                total = sum(counts.values())
                if a.preview_every and total % a.preview_every == 1:
                    cv2.imwrite(os.path.join(a.out, "previews", name + ".jpg"), preview(img, lines))
                if total % 200 == 0:
                    el = time.time() - t_start
                    print(f"[gen] {total} frames ({counts}) {total / el:.1f} fps, map={m}", flush=True)
                if saved >= target:
                    break
            ep += 1
    meta_f.close()
    for env in envs.values():
        env.close()
    print(f"[gen] done: {counts} in {time.time() - t_start:.0f}s -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
