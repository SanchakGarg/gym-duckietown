"""Ground-plane (extrinsic) calibration -> image-to-ground homography in the robot YAML.

Lay the checkerboard flat on the floor in front of the robot, its rows parallel to the
wheel axle, and measure where its corner is (robot frame: origin = middle of the wheel axle
on the floor, x forward, y left):

    --x0  forward distance (m) from the axle to the *nearest* row of inner corners
    --y0  lateral position (m) of the *right-most* column of inner corners (negative = right)

Defaults correspond to the Duckietown extrinsic-calibration setup (7x5 inner corners, 31 mm
squares, board centred in front of the robot): measure and override them for your setup.

    python -m duckie_yolo.robot.calibrate_extrinsics --host <robot> --robot-config configs/robot_mybot.yaml
    python -m duckie_yolo.robot.calibrate_extrinsics --image floor.png --robot-config ...

Run calibrate_intrinsics first: the homography is computed on the rectified image.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from ..perception.camera_model import CameraModel
from .calibrate_intrinsics import find_board
from .robot_config import RobotConfig


def order_grid(corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reshape detected corners to (rows, cols, 2) with row 0 nearest the robot (largest v)
    and column 0 right-most (largest u)."""
    g = corners.reshape(-1, 2)
    if g.shape[0] != rows * cols:
        raise ValueError("unexpected number of corners")
    g = g.reshape(rows, cols, 2)
    if g[0, :, 1].mean() < g[-1, :, 1].mean():
        g = g[::-1]
    if g[:, 0, 0].mean() < g[:, -1, 0].mean():
        g = g[:, ::-1]
    return g


def ground_grid(rows: int, cols: int, square: float, x0: float, y0: float) -> np.ndarray:
    i, j = np.mgrid[0:rows, 0:cols]
    return np.stack([x0 + i * square, y0 + j * square], axis=-1).astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--image", default="", help="use this raw camera image instead of grabbing one")
    ap.add_argument("--robot-config", default="configs/robot_duckiebot.yaml")
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=0.031)
    ap.add_argument("--x0", type=float, default=0.191)
    ap.add_argument("--y0", type=float, default=-0.093)
    a = ap.parse_args(argv)

    robot = RobotConfig.load(a.robot_config)
    cam = robot.camera_model()
    if not robot.calibrated.get("intrinsics"):
        print("[extrinsics] WARNING: intrinsics not calibrated, using default rectification")
    if a.image:
        raw = cv2.imread(a.image)
    else:
        from .real_env import RealRobotEnv

        env = RealRobotEnv(a.host, a.port, encoding="jpeg", jpeg_quality=95)
        env.reset()
        raw = cv2.cvtColor(env.wheels(0.0, 0.0), cv2.COLOR_RGB2BGR)
        env.close()
    rect = cam.rectify(raw)
    corners = find_board(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (a.cols, a.rows))
    if corners is None:
        cv2.imwrite("extrinsics_failed.jpg", rect)
        raise SystemExit("checkerboard not found in the rectified image (saved extrinsics_failed.jpg)")
    img_g = order_grid(corners, a.rows, a.cols)
    gnd_g = ground_grid(a.rows, a.cols, a.square, a.x0, a.y0)
    H, _ = cv2.findHomography(img_g.reshape(-1, 2), gnd_g.reshape(-1, 2), 0)

    new_cam = CameraModel(rect.shape[1], rect.shape[0], H, cam.rectifier)
    proj, _ = new_cam.image_to_ground(img_g.reshape(-1, 2))
    err = np.linalg.norm(proj - gnd_g.reshape(-1, 2), axis=1)
    print(f"[extrinsics] reprojection error on the floor: mean {1000 * err.mean():.1f} mm, max {1000 * err.max():.1f} mm")

    # visual check: draw a 10 cm ground grid and the robot's straight-ahead line
    vis = rect.copy()
    for x in np.arange(0.1, 0.81, 0.1):
        uv, ok = new_cam.ground_to_image(np.array([[x, y] for y in np.linspace(-0.3, 0.3, 30)]))
        cv2.polylines(vis, [uv[ok].astype(np.int32)], False, (0, 255, 0), 1)
    for y in np.arange(-0.3, 0.31, 0.1):
        uv, ok = new_cam.ground_to_image(np.array([[x, y] for x in np.linspace(0.1, 0.8, 30)]))
        cv2.polylines(vis, [uv[ok].astype(np.int32)], False, (0, 0, 255) if abs(y) < 1e-6 else (0, 255, 0), 1)
    cv2.drawChessboardCorners(vis, (a.cols, a.rows), corners, True)
    out_img = os.path.splitext(a.robot_config)[0] + "_extrinsics_check.jpg"
    cv2.imwrite(out_img, vis)

    robot.camera = new_cam.to_dict()
    robot.calibrated["extrinsics"] = True
    robot.save(a.robot_config)
    print(f"[extrinsics] wrote {a.robot_config}; check the grid overlay in {out_img}")


if __name__ == "__main__":
    main()
