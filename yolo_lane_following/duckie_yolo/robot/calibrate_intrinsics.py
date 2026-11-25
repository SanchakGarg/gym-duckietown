"""Camera intrinsic calibration (OpenCV only) -> rectifier in the robot calibration YAML.

1. Print a checkerboard (default 7x5 *inner* corners, 31 mm squares - the Duckietown board).
2. Capture 20-40 views at different distances/angles/positions, covering the image corners:
      python -m duckie_yolo.robot.calibrate_intrinsics --host <robot> --capture calib/intr --count 30
   (saves a frame whenever the board is detected and the view changed enough; move the board
   slowly in front of the robot), or put your own images in a folder.
3. Calibrate and write the config:
      python -m duckie_yolo.robot.calibrate_intrinsics --images calib/intr --robot-config configs/robot_mybot.yaml

The raw image is rectified to a virtual pinhole camera matching the simulator camera
(640x480, 75 deg vertical FOV) so that the YOLO model sees sim-like geometry.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..perception.camera_model import Rectifier
from .robot_config import RobotConfig


def find_board(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)


class ViewSelector:
    """Decides when a detected checkerboard view is new enough to keep (moved >= min_move px, or
    the board has been held still for longer than hold_s seconds)."""

    def __init__(self, min_move: float = 40.0, hold_s: float = 3.0):
        self.min_move, self.hold_s = min_move, hold_s
        self.last_center = None
        self.t_last = 0.0

    def accept(self, corners: np.ndarray, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        c = corners.reshape(-1, 2).mean(0)
        if self.last_center is not None and np.linalg.norm(c - self.last_center) < self.min_move and now - self.t_last < self.hold_s:
            return False
        self.last_center, self.t_last = c, now
        return True


def capture(args):
    from .real_env import RealRobotEnv

    os.makedirs(args.capture, exist_ok=True)
    env = RealRobotEnv(args.host, args.port, encoding="jpeg", jpeg_quality=95)
    env.reset()
    saved = 0
    selector = ViewSelector()
    pattern = (args.cols, args.rows)
    print(f"[intrinsics] capturing {args.count} views to {args.capture} (move the board around)")
    try:
        while saved < args.count:
            rgb = env.wheels(0.0, 0.0)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            corners = find_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), pattern)
            if corners is None or not selector.accept(corners):
                continue
            cv2.imwrite(os.path.join(args.capture, f"intr_{saved:03d}.png"), bgr)
            saved += 1
            c = corners.reshape(-1, 2).mean(0)
            spread = np.ptp(corners.reshape(-1, 2), axis=0).mean()
            print(f"  saved {saved}/{args.count} (board center {c.astype(int)}, size {spread:.0f}px)")
    finally:
        env.close()


def image_files(folder: str) -> List[str]:
    return sorted(sum((glob.glob(os.path.join(folder, e)) for e in ("*.png", "*.jpg")), []))


def calibrate_images(
    files: Sequence[str],
    robot_config: str,
    cols: int = 7,
    rows: int = 5,
    square: float = 0.031,
    model: str = "fisheye",
    fov: float = 75.0,
    out_size: Tuple[int, int] = (640, 480),
) -> Dict[str, Any]:
    """Calibrate from checkerboard images and write the rectifier into ``robot_config``.

    Returns {"rms", "views", "files", "K", "D", "sample"} where sample is the path of a raw|rectified
    side-by-side image. Raises ValueError when fewer than 8 views contain the board.
    """
    pattern = (cols, rows)
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    obj_pts, img_pts, size = [], [], None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)
        corners = find_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
        if corners is not None:
            obj_pts.append(objp.copy())
            img_pts.append(corners.reshape(1, -1, 2).astype(np.float64))
    if len(img_pts) < 8:
        raise ValueError(f"need at least 8 views with the board, found {len(img_pts)}/{len(files)}")

    if model == "fisheye":
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj_pts, img_pts, size, K, D, flags=flags, criteria=crit)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(
            [o.reshape(-1, 3).astype(np.float32) for o in obj_pts],
            [i.reshape(-1, 2).astype(np.float32) for i in img_pts],
            size, None, None,
        )

    out_size = tuple(out_size)
    rect = Rectifier(K, D, size, out_size, Rectifier.virtual_pinhole(out_size, fov), model=model)
    robot = RobotConfig.load(robot_config)
    cam_d = robot.camera_model().to_dict()
    cam_d.update(width=out_size[0], height=out_size[1], rectifier=rect.to_dict())
    robot.camera = cam_d
    robot.calibrated["intrinsics"] = True
    robot.calibrated["extrinsics"] = False  # the homography lives in the rectified image: redo it
    robot.save(robot_config)
    sample_path = os.path.splitext(robot_config)[0] + "_rectified_sample.jpg"
    sample = cv2.imread(files[0])
    cv2.imwrite(sample_path, np.hstack([cv2.resize(sample, out_size), rect(sample)]))
    return {"rms": float(rms), "views": len(img_pts), "files": len(files), "K": K, "D": D, "sample": sample_path}


def calibrate(args):
    files = image_files(args.images)
    try:
        r = calibrate_images(files, args.robot_config, args.cols, args.rows, args.square, args.model, args.fov,
                             (args.out_width, args.out_height))
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"[intrinsics] board found in {r['views']}/{r['files']} images")
    print(f"[intrinsics] model={args.model} RMS reprojection error = {r['rms']:.3f} px")
    print("K =\n", r["K"], "\nD =", r["D"].ravel())
    print(f"[intrinsics] wrote {args.robot_config} - now run calibrate_extrinsics")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--capture", default="", help="capture views from the robot into this folder")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--images", default="", help="folder with checkerboard images to calibrate from")
    ap.add_argument("--robot-config", default="configs/robot_duckiebot.yaml")
    ap.add_argument("--cols", type=int, default=7, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=5, help="inner corners per column")
    ap.add_argument("--square", type=float, default=0.031, help="square size (m)")
    ap.add_argument("--model", default="fisheye", choices=["fisheye", "pinhole"])
    ap.add_argument("--fov", type=float, default=75.0, help="vertical FOV of the rectified virtual camera")
    ap.add_argument("--out-width", type=int, default=640)
    ap.add_argument("--out-height", type=int, default=480)
    a = ap.parse_args(argv)
    if a.capture:
        capture(a)
        if not a.images:
            a.images = a.capture
    if a.images:
        calibrate(a)
    elif not a.capture:
        ap.error("give --capture and/or --images")


if __name__ == "__main__":
    main()
