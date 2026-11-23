"""Thin helpers around gym-duckietown (imported lazily so the core never needs it)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# Maps used by default. All ship with duckietown-world-daffy.
TRAIN_MAPS = [
    "loop_empty",
    "small_loop",
    "small_loop_cw",
    "zigzag_dists",
    "udem1",
    "4way",
    "straight_road",
    "loop_obstacles",
    "loop_pedestrians",
    "loop_dyn_duckiebots",
    "regress_4way_adam",
    "ETHZ_loop",
    "TTIC_ripltown",
    "robotarium1",
    "Montreal_loop",
    "ETH_small_intersect",
]
# never used for training: generalization check
HELDOUT_MAPS = ["ETHZ_autolab_technical_track", "ETH_large_intersect"]

# Steady-state response of gym-duckietown (daffy) motor dynamics to DuckietownEnv [v, omega]
# commands, measured with tools/identify_sim_dynamics.py: realized v = 0.699 v_cmd,
# realized omega = 0.445 omega_cmd (time to 90%: ~0.63 s).
SIM_V_RATIO = 0.699
SIM_OMEGA_RATIO = 0.445


def available_maps():
    from duckietown_world.resources import list_maps2

    return sorted(list_maps2().keys())


def make_env(
    map_name: str,
    seed: int = 0,
    domain_rand: bool = True,
    camera_rand: bool = False,
    dynamics_rand: bool = False,
    style: str = "photos",
    max_steps: int = 1500,
    width: int = 640,
    height: int = 480,
    accept_start_angle_deg: float = 45,
    user_tile_start=None,
    draw_curve: bool = False,
):
    from gym_duckietown.envs import DuckietownEnv

    return DuckietownEnv(
        map_name=map_name,
        seed=seed,
        domain_rand=domain_rand,
        camera_rand=camera_rand,
        dynamics_rand=dynamics_rand,
        style=style,
        max_steps=max_steps,
        camera_width=width,
        camera_height=height,
        accept_start_angle_deg=accept_start_angle_deg,
        user_tile_start=user_tile_start,
        draw_curve=draw_curve,
        distortion=False,
    )


def gt_lane_pose(env) -> Optional[Tuple[float, float]]:
    """Ground-truth (d, phi) from the simulator, or None when not in a lane.

    Same convention as duckie_yolo.estimation: d > 0 robot right of center, phi > 0 robot
    heading to the right of the lane direction.
    """
    from gym_duckietown.simulator import NotInLane

    u = env.unwrapped
    try:
        lp = u.get_lane_pos2(u.cur_pos, u.cur_angle)
    except NotInLane:
        return None
    return float(lp.dist), float(lp.angle_rad)


def sim_time(env) -> float:
    u = env.unwrapped
    return float(u.step_count) * float(u.delta_time)


def set_pose(env, pos_xz: Tuple[float, float], angle: float) -> None:
    """Teleport the robot (sim frame: x, z on the floor; angle around the up axis)."""
    u = env.unwrapped
    u.cur_pos = np.array([pos_xz[0], 0.0, pos_xz[1]])
    u.cur_angle = float(angle)


def lane_frame_pose(env, d: float, phi: float, along: float = 0.0) -> Tuple[Tuple[float, float], float]:
    """Pose with a given (d, phi) relative to the lane curve at the robot's current location."""
    u = env.unwrapped
    point, tangent = u.closest_curve_point(u.cur_pos, u.cur_angle)
    if point is None:
        raise RuntimeError("robot is not in a lane")
    t = tangent / np.linalg.norm(tangent)
    right = np.cross(t, np.array([0.0, 1.0, 0.0]))  # == rightVec in Simulator.get_lane_pos2
    p = point + along * t + d * right
    lane_angle = math.atan2(-t[2], t[0])  # inverse of get_dir_vec: (cos a, 0, -sin a)
    return (float(p[0]), float(p[2])), lane_angle - phi
