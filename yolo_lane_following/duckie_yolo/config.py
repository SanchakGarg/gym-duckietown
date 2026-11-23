"""Typed configuration for the lane-following agent (loadable from / dumpable to YAML)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


@dataclass
class LaneGeometry:
    # Lateral offsets of the lane *center* (where we want to drive) from the painted lines,
    # measured perpendicular to the lines, in meters.
    # gym-duckietown follows a curve 0.20 * tile_size (0.585 m) = 0.117 m right of the yellow
    # center line; the white edge-line center is ~0.15 m to the right of that curve.
    yellow_offset: float = 0.117
    white_offset: float = 0.150


@dataclass
class EstimatorConfig:
    work_scale: float = 0.5  # masks are rasterized at this fraction of the image size
    stride: int = 2  # pixel sub-sampling stride inside masks (at work resolution)
    max_points: int = 600  # per class, after sub-sampling
    x_min: float = 0.10  # ground region of interest (m, robot frame)
    x_max: float = 0.55
    y_abs_max: float = 0.60
    min_points: int = 15  # minimum points to fit a line marking
    poly_degree: int = 2
    ransac_iters: int = 40
    ransac_thresh: float = 0.025  # m
    min_x_span_quadratic: float = 0.18  # fall back to a straight-line fit for short spans
    quadratic_gain: float = 1.5  # use a quadratic only if line-RMS > gain * quad-RMS + 2 mm
    white_min_right_of_yellow: float = 0.06  # drop white points not clearly right of the yellow line
    max_center_disagreement: float = 0.09  # m, yellow- vs white-derived center at x=0.3 m
    ema_alpha: float = 0.6  # weight of the new measurement in the temporal filter
    # stop line / obstacles
    stopline_corridor: float = 0.22  # |y - lane_center| (m) for red points to count
    min_red_points: int = 12
    obstacle_min_area_px: int = 30  # at work resolution


@dataclass
class ControllerConfig:
    v_ref: float = 0.35  # nominal forward speed command
    v_min: float = 0.12
    lookahead: float = 0.30  # m
    curvature_gain: float = 0.20  # v = v_ref / (1 + curvature_gain * |kappa|)
    omega_max: float = 6.0
    lost_timeout: float = 0.5  # s without a valid lane before stopping
    lost_speed_scale: float = 0.5  # keep the last command at this speed scale while briefly lost
    min_confidence: float = 0.15


@dataclass
class BehaviorConfig:
    enable_stoplines: bool = True
    stopline_trigger_x: float = 0.22  # stop when the red line is closer than this (m ahead of axle)
    stop_duration: float = 2.0  # s
    cross_duration: float = 2.2  # s of driving straight after the stop
    cross_speed: float = 0.25
    stopline_cooldown: float = 3.0  # s during which red lines are ignored after crossing
    cross_actuator_tau: float = 0.27  # s, actuator lag assumed when dead-reckoning through intersections
    enable_obstacles: bool = True
    obstacle_trigger_x: float = 0.40  # m
    obstacle_lane_margin: float = 0.16  # |y_obstacle - lane_center| below which it blocks us
    obstacle_clear_time: float = 0.6  # s


@dataclass
class SegmenterConfig:
    weights: str = ""  # .pt / .onnx / .engine
    imgsz: int = 320
    conf: float = 0.35
    iou: float = 0.5
    device: str = ""  # "" = auto (cuda:0 / mps / cpu)
    half: bool = False  # informational: FP16 comes from the TensorRT engine export


@dataclass
class ActuationConfig:
    """Platform command gains applied to the final (v, omega) command.

    The controller reasons in physical units; real platforms respond with their own steady-state
    gains (gym-duckietown's identified motor dynamics: realized v = 0.699 v_cmd and realized
    omega = 0.445 omega_cmd, see tools/identify_sim_dynamics.py). Set v_gain = 1/v_ratio and
    omega_gain = 1/omega_ratio. Real robot: calibrate_wheels.py (--distance / --rotate).
    """

    v_gain: float = 1.0
    omega_gain: float = 1.0


@dataclass
class AgentConfig:
    lane: LaneGeometry = field(default_factory=LaneGeometry)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)


# ---------------------------------------------------------------------- helpers
def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively build a dataclass from a (possibly partial) dict; unknown keys raise."""
    if data is None:
        return cls()
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    defaults = cls()
    for name, f in names.items():
        if name not in data:
            continue
        cur = getattr(defaults, name)
        if dataclasses.is_dataclass(cur):
            kwargs[name] = from_dict(type(cur), data[name])
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def to_dict(obj) -> Dict[str, Any]:
    return dataclasses.asdict(obj)


def load_agent_config(path: str = "") -> AgentConfig:
    if not path:
        return AgentConfig()
    import yaml

    with open(path) as f:
        return from_dict(AgentConfig, yaml.safe_load(f) or {})


def save_yaml(obj, path: str) -> None:
    import yaml

    data = to_dict(obj) if dataclasses.is_dataclass(obj) else obj
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
