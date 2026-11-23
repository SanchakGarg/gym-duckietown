"""Per-robot calibration file (camera rectification + ground homography + wheel trim).

Created / updated by the calibration scripts:
    calibrate_intrinsics.py  -> camera.rectifier
    calibrate_extrinsics.py  -> camera.H_img2ground
    calibrate_wheels.py      -> wheels.gain / wheels.trim
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import ActuationConfig
from ..control.kinematics import WheelCalibration
from ..perception.camera_model import CameraModel, Rectifier

# Placeholder intrinsics of the stock Duckiebot camera at 640x480 (plumb-bob model), from the
# Duckietown default calibration also used by gym-duckietown's UndistortWrapper. Calibrate
# your own camera: the DB21/DB26 160-degree lens differs noticeably.
DEFAULT_K = [[305.5718893575089, 0.0, 303.0797142544728], [0.0, 308.8338858195428, 231.8845403702499], [0.0, 0.0, 1.0]]
DEFAULT_D = [-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0.0]


@dataclass
class RobotConfig:
    name: str = "duckiebot"
    camera: dict = field(default_factory=dict)  # CameraModel.to_dict()
    wheels: WheelCalibration = field(default_factory=WheelCalibration)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    calibrated: dict = field(default_factory=lambda: {"intrinsics": False, "extrinsics": False, "wheels": False})

    def camera_model(self) -> CameraModel:
        if self.camera:
            return CameraModel.from_dict(self.camera)
        return default_camera_model()

    def save(self, path: str) -> None:
        import yaml

        d = {
            "name": self.name,
            "calibrated": self.calibrated,
            "wheels": dataclasses.asdict(self.wheels),
            "actuation": dataclasses.asdict(self.actuation),
            "camera": self.camera or default_camera_model().to_dict(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path: Optional[str]) -> "RobotConfig":
        if not path or not os.path.exists(path):
            return cls()
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(
            name=d.get("name", "duckiebot"),
            camera=d.get("camera") or {},
            wheels=WheelCalibration(**(d.get("wheels") or {})),
            actuation=ActuationConfig(**(d.get("actuation") or {})),
            calibrated=d.get("calibrated") or {"intrinsics": False, "extrinsics": False, "wheels": False},
        )


def default_camera_model(width: int = 640, height: int = 480) -> CameraModel:
    """Uncalibrated guess: stock intrinsics, sim-like mounting (≈10.8 cm high, ≈19° tilt)."""
    rect = Rectifier(
        np.array(DEFAULT_K), np.array(DEFAULT_D), (width, height), (width, height),
        Rectifier.virtual_pinhole((width, height)), model="pinhole",
    )
    cm = CameraModel.sim_default(width, height)
    return CameraModel(width, height, cm.H, rect)
