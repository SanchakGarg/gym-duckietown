"""Camera geometry: (optional) fisheye rectification + image <-> ground-plane homography.

Robot ground frame used everywhere in duckie_yolo:
    origin = midpoint of the wheel axle projected on the floor
    x      = forward (m), y = left (m), z = up (m)

All homographies act on pixels of the *rectified* image. In simulation the image is
already an ideal pinhole image, so rectification is the identity. On the real robot the
wide-angle IMX219 image is remapped to a virtual pinhole camera whose intrinsics match the
simulator camera, so the YOLO model sees the same geometry in both worlds.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# gym-duckietown camera constants (src/gym_duckietown/simulator.py)
SIM_CAMERA_FOV_Y_DEG = 75.0
SIM_CAMERA_HEIGHT = 0.108  # CAMERA_FLOOR_DIST
SIM_CAMERA_FORWARD = 0.066  # CAMERA_FORWARD_DIST (camera is ahead of the wheel axle)
SIM_CAMERA_PITCH_DEG = 19.15  # CAMERA_ANGLE (tilted down)


def pinhole_K(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    """Intrinsics of an ideal pinhole camera with square pixels (like gluPerspective)."""
    f = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def camera_rotation(pitch_deg: float, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Rotation R_cr mapping robot-frame vectors to camera-frame vectors.

    Camera frame: x right, y down, z forward (OpenCV convention).
    pitch > 0 tilts the camera down towards the floor.
    """
    p = math.radians(pitch_deg)
    # nominal camera axes expressed in the robot frame
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)  # = (-sin p, 0, -cos p)
    R = np.stack([right, down, fwd])  # rows: camera axes in robot coords
    if yaw_deg or roll_deg:
        y = math.radians(yaw_deg)
        r = math.radians(roll_deg)
        # yaw about the camera's down axis, roll about its forward axis (small corrections)
        Ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
        Rr = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
        R = Rr @ Ry @ R
    return R


def ground_to_image_homography(K: np.ndarray, R_cr: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    """3x3 homography G with  s*[u, v, 1]^T = G [x, y, 1]^T  for floor points (x, y, 0)."""
    t = -R_cr @ np.asarray(cam_pos, dtype=float)
    G = K @ np.column_stack([R_cr[:, 0], R_cr[:, 1], t])
    # positive scaling only: the sign of the homogeneous coordinate encodes "in front of the camera"
    return G / np.linalg.norm(G)


class CameraModel:
    """Maps rectified-image pixels to robot ground coordinates and back."""

    def __init__(self, width: int, height: int, H_img2ground: np.ndarray, rectifier: Optional["Rectifier"] = None):
        self.width = int(width)
        self.height = int(height)
        H = np.asarray(H_img2ground, dtype=float)
        H = H / np.linalg.norm(H)
        # A homography is only defined up to scale, *including sign* (cv2.findHomography normalises
        # H[2,2] = 1). Fix the sign so that the bottom-centre pixel, which always sees the floor
        # in front of the robot, maps with a positive homogeneous coordinate.
        if H[2] @ np.array([self.width / 2.0, self.height - 1.0, 1.0]) < 0:
            H = -H
        self.H = H
        G = np.linalg.inv(H)
        self.G = G / np.linalg.norm(G)
        self.rectifier = rectifier

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_pose(
        cls,
        width: int,
        height: int,
        fov_y_deg: float,
        cam_height: float,
        cam_forward: float,
        pitch_deg: float,
        cam_left: float = 0.0,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        rectifier: Optional["Rectifier"] = None,
    ) -> "CameraModel":
        K = pinhole_K(width, height, fov_y_deg)
        R = camera_rotation(pitch_deg, yaw_deg, roll_deg)
        G = ground_to_image_homography(K, R, np.array([cam_forward, cam_left, cam_height]))
        return cls(width, height, np.linalg.inv(G), rectifier)

    @classmethod
    def sim_default(cls, width: int = 640, height: int = 480) -> "CameraModel":
        """Nominal gym-duckietown camera (no camera randomization)."""
        return cls.from_pose(
            width,
            height,
            SIM_CAMERA_FOV_Y_DEG,
            SIM_CAMERA_HEIGHT,
            SIM_CAMERA_FORWARD,
            SIM_CAMERA_PITCH_DEG,
        )

    # ------------------------------------------------------------------ geometry
    def rectify(self, img: np.ndarray) -> np.ndarray:
        if self.rectifier is None:
            return img
        return self.rectifier(img)

    def image_to_ground(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project pixels (N,2) [u=column, v=row] to floor points (N,2) [x fwd, y left].

        Returns (points, valid). Pixels at/above the horizon are invalid.
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        homog = np.column_stack([uv, np.ones(len(uv))]) @ self.H.T
        w = homog[:, 2]
        valid = w > 1e-9
        pts = np.full((len(uv), 2), np.nan)
        pts[valid] = homog[valid, :2] / w[valid, None]
        return pts, valid

    def ground_to_image(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project floor points (N,2) to pixels (N,2). Points behind the camera are invalid."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        homog = np.column_stack([xy, np.ones(len(xy))]) @ self.G.T
        s = homog[:, 2]
        valid = s > 1e-9
        uv = np.full((len(xy), 2), np.nan)
        uv[valid] = homog[valid, :2] / s[valid, None]
        return uv, valid

    def horizon_row(self) -> float:
        """Image row of the horizon (rows above it never see the floor)."""
        # The horizon is where the third row of H vanishes: H[2] . (u, v, 1) = 0  (take u = center)
        h = self.H[2]
        if abs(h[1]) < 1e-12:
            return 0.0
        return float(-(h[0] * self.width / 2.0 + h[2]) / h[1])

    # ------------------------------------------------------------------ (de)serialization
    def to_dict(self) -> dict:
        d = {"width": self.width, "height": self.height, "H_img2ground": self.H.tolist()}
        if self.rectifier is not None:
            d["rectifier"] = self.rectifier.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        rect = Rectifier.from_dict(d["rectifier"]) if d.get("rectifier") else None
        return cls(d["width"], d["height"], np.array(d["H_img2ground"]), rect)


class Rectifier:
    """Undistort a raw (fisheye or plumb-bob) camera image into a virtual pinhole image."""

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        in_size: Tuple[int, int],
        out_size: Tuple[int, int],
        K_new: np.ndarray,
        model: str = "fisheye",
    ):
        import cv2  # local import keeps the module importable without OpenCV for pure geometry

        self.K = np.asarray(K, dtype=float)
        self.D = np.asarray(D, dtype=float).reshape(-1)
        self.in_size = (int(in_size[0]), int(in_size[1]))
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.K_new = np.asarray(K_new, dtype=float)
        self.model = model
        if model == "fisheye":
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D[:4], np.eye(3), self.K_new, self.out_size, cv2.CV_16SC2
            )
        elif model == "pinhole":
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K_new, self.out_size, cv2.CV_16SC2
            )
        else:
            raise ValueError(f"unknown camera model {model!r}")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if (img.shape[1], img.shape[0]) != self.in_size:
            img = cv2.resize(img, self.in_size, interpolation=cv2.INTER_AREA)
        return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def virtual_pinhole(out_size: Tuple[int, int] = (640, 480), fov_y_deg: float = SIM_CAMERA_FOV_Y_DEG) -> np.ndarray:
        return pinhole_K(out_size[0], out_size[1], fov_y_deg)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "in_size": list(self.in_size),
            "out_size": list(self.out_size),
            "K_new": self.K_new.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rectifier":
        return cls(
            np.array(d["K"]),
            np.array(d["D"]),
            tuple(d["in_size"]),
            tuple(d["out_size"]),
            np.array(d["K_new"]),
            d.get("model", "fisheye"),
        )
