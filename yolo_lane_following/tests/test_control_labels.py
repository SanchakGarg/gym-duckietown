import math

import numpy as np

from duckie_yolo.agent import LaneFollowingAgent
from duckie_yolo.classes import DUCKIE, RED, WHITE, YELLOW
from duckie_yolo.config import AgentConfig, BehaviorConfig, ControllerConfig, from_dict, to_dict
from duckie_yolo.control.behavior import BehaviorFSM, State
from duckie_yolo.control.kinematics import WheelCalibration, twist_to_wheels
from duckie_yolo.control.pure_pursuit import PurePursuitController
from duckie_yolo.estimation.lane_estimator import LaneEstimate
from duckie_yolo.perception.camera_model import CameraModel
from duckie_yolo.perception.segmenters import segmentation_from_label_map
from duckie_yolo.sim.labels import (
    gen_segmentation_color,
    label_map_to_polygons,
    object_palette,
    polygons_to_yolo_lines,
    seg_to_label_map,
    yolo_lines_to_label_map,
)

from test_geometry_estimator import render_lane


def reference_duckietown_env_step(action, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, baseline=0.102):
    """Verbatim math of gym_duckietown.envs.DuckietownEnv.step (without the simulator)."""
    vel, angle = action
    k_r_inv = (gain + trim) / k
    k_l_inv = (gain - trim) / k
    omega_r = (vel + 0.5 * angle * baseline) / radius
    omega_l = (vel - 0.5 * angle * baseline) / radius
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv
    return max(min(u_l, limit), -limit), max(min(u_r, limit), -limit)


def test_kinematics_matches_duckietown_env():
    for v, w in [(0.3, 0.0), (0.35, 2.0), (0.2, -4.0), (-0.2, 1.0), (1.0, 10.0)]:
        for trim in (0.0, 0.05):
            ref = reference_duckietown_env_step((v, w), trim=trim)
            got = twist_to_wheels(v, w, WheelCalibration(trim=trim))
            np.testing.assert_allclose(got, ref)


def test_pure_pursuit_steers_towards_target():
    pp = PurePursuitController(ControllerConfig())
    left = LaneEstimate(valid=True, lookahead=(0.28, 0.1), confidence=1.0)
    right = LaneEstimate(valid=True, lookahead=(0.28, -0.1), confidence=1.0)
    straight = LaneEstimate(valid=True, lookahead=(0.3, 0.0), confidence=1.0)
    assert pp(left)[1] > 0 > pp(right)[1]
    v, w = pp(straight)
    assert abs(w) < 1e-9 and abs(v - ControllerConfig().v_ref) < 1e-9
    assert pp(left)[0] < v  # slows down in curves
    assert pp(LaneEstimate())[0] == 0.0


def test_behavior_stop_line_cycle():
    cfg = BehaviorConfig(stop_duration=2.0, cross_duration=1.0, stopline_cooldown=3.0)
    fsm = BehaviorFSM(cfg)
    lane = LaneEstimate(valid=True, center=np.array([0.0, 0.0, 0.0]))
    near_red = LaneEstimate(valid=True, center=np.array([0.0, 0.0, 0.0]), stopline_x=0.18)
    assert fsm.update(lane, (0.3, 0.1), 0.0, False) == (0.3, 0.1, State.DRIVE)
    assert fsm.update(near_red, (0.3, 0.1), 1.0, False)[2] == State.STOP_LINE_WAIT
    assert fsm.update(near_red, (0.3, 0.1), 2.5, False)[:2] == (0.0, 0.0)
    v, w, s = fsm.update(near_red, (0.3, 0.1), 3.1, False)
    assert s == State.CROSS and v == cfg.cross_speed and w == 0.0
    assert fsm.update(near_red, (0.3, 0.1), 4.2, False)[2] == State.DRIVE  # cooldown ignores the same line
    assert fsm.update(near_red, (0.3, 0.1), 7.5, False)[2] == State.STOP_LINE_WAIT  # a new line later


def test_behavior_obstacle_and_lost():
    fsm = BehaviorFSM(BehaviorConfig(obstacle_clear_time=0.5))
    center = np.array([0.0, 0.0, 0.0])
    blocked = LaneEstimate(valid=True, center=center, obstacles=[(DUCKIE, 0.3, 0.02)])
    beside = LaneEstimate(valid=True, center=center, obstacles=[(DUCKIE, 0.3, 0.35)])
    free = LaneEstimate(valid=True, center=center)
    assert fsm.update(beside, (0.3, 0.0), 0.0, False)[2] == State.DRIVE
    assert fsm.update(blocked, (0.3, 0.0), 0.1, False) == (0.0, 0.0, State.OBSTACLE_STOP)
    assert fsm.update(free, (0.3, 0.0), 0.3, False)[2] == State.OBSTACLE_STOP
    assert fsm.update(free, (0.3, 0.0), 0.7, False)[2] == State.DRIVE
    assert fsm.update(free, (0.3, 0.0), 0.8, True) == (0.0, 0.0, State.LOST)
    off = BehaviorFSM(BehaviorConfig(enable_obstacles=False))
    assert off.update(blocked, (0.3, 0.0), 0.0, False)[2] == State.DRIVE


def test_agent_end_to_end_with_label_map():
    cfg = AgentConfig()
    agent = LaneFollowingAgent(None, CameraModel.sim_default(), cfg)
    rgb = np.zeros((480, 640, 3), np.uint8)
    # robot left of center and heading left -> must steer right
    lm = render_lane(-0.05, -0.15)
    step = agent.act(rgb, t=0.0, segmentation=segmentation_from_label_map(lm, cfg.estimator.work_scale))
    assert step.state == State.DRIVE
    assert step.action[0] > 0 and step.action[1] < 0
    # nothing visible for longer than lost_timeout -> stop
    empty = segmentation_from_label_map(np.full((480, 640), -1, np.int16), cfg.estimator.work_scale)
    agent.act(rgb, t=0.1, segmentation=empty)
    step = agent.act(rgb, t=1.0, segmentation=empty)
    assert step.state == State.LOST and np.allclose(step.action, 0)


def test_config_roundtrip():
    cfg = AgentConfig()
    cfg.controller.v_ref = 0.2
    cfg.behavior.enable_obstacles = False
    back = from_dict(AgentConfig, to_dict(cfg))
    assert back == cfg
    partial = from_dict(AgentConfig, {"controller": {"lookahead": 0.4}})
    assert partial.controller.lookahead == 0.4 and partial.controller.v_ref == ControllerConfig().v_ref


def test_segmentation_color_hash_matches_simulator():
    # values observed in gym-duckietown segment renders
    assert gen_segmentation_color("duckie") == (100, 117, 226)
    assert gen_segmentation_color("duckiebot") == (100, 117, 226)  # same colour -> split by re-rendering
    assert gen_segmentation_color("cone") == (226, 111, 101)
    assert gen_segmentation_color("barrier") == (224, 201, 156)
    assert gen_segmentation_color("bus") == (216, 171, 15)


def test_seg_to_label_map_classes():
    seg = np.zeros((120, 160, 3), np.uint8)
    seg[:20] = (255, 0, 255)  # sky -> background
    seg[40:60, 10:20] = (238, 222, 110)  # pale photo-texture yellow dash
    seg[40:60, 30:40] = (245, 244, 242)  # white line
    seg[70:75, 10:60] = (240, 30, 35)  # red stop line
    seg[80:100, 100:120] = (100, 117, 226)  # duckie
    seg[80:100, 130:150] = (216, 171, 15)  # bus: yellow-ish but must NOT be a lane marking
    seg[100:110, 10:30] = (226, 111, 101)  # cone: red-ish but must NOT be a stop line
    lm = seg_to_label_map(seg, object_palette(["duckie", "bus", "cone"]))
    assert (lm[45, 15], lm[45, 35], lm[72, 30], lm[90, 110]) == (YELLOW, WHITE, RED, DUCKIE)
    assert lm[10, 10] == -1 and lm[90, 140] == -1 and lm[105, 20] == -1 and lm[110, 80] == -1


def test_yolo_polygon_roundtrip():
    lm = np.full((120, 160), -1, np.int16)
    lm[40:80, 20:40] = YELLOW
    lm[10:100, 100:110] = WHITE
    polys = label_map_to_polygons(lm, min_area=10)
    assert sorted(c for c, _ in polys) == [YELLOW, WHITE]
    lines = polygons_to_yolo_lines(polys, 160, 120)
    back = yolo_lines_to_label_map(lines, 160, 120)
    for c in (YELLOW, WHITE):
        a, b = lm == c, back == c
        assert (a & b).sum() / (a | b).sum() > 0.85


def test_actuation_gains_scale_command():
    from duckie_yolo.config import ActuationConfig

    cfg = AgentConfig()
    base = LaneFollowingAgent(None, CameraModel.sim_default(), cfg)
    cfg2 = AgentConfig()
    cfg2.actuation = ActuationConfig(v_gain=2.0, omega_gain=3.0)
    scaled = LaneFollowingAgent(None, CameraModel.sim_default(), cfg2)
    seg = segmentation_from_label_map(render_lane(-0.05, -0.15), cfg.estimator.work_scale)
    rgb = np.zeros((480, 640, 3), np.uint8)
    a = base.act(rgb, t=0.0, segmentation=seg).action
    b = scaled.act(rgb, t=0.0, segmentation=seg).action
    np.testing.assert_allclose(b, a * np.array([2.0, 3.0]))


def test_yolo_result_to_masks_without_weights():
    """YoloSegmenter.from_result on a stand-in for an Ultralytics Results object."""
    from types import SimpleNamespace

    from duckie_yolo.perception.segmenters import YoloSegmenter

    class T:  # minimal tensor stand-in
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    poly_y = np.array([[100, 300], [140, 300], [140, 400], [100, 400]], np.float32)
    poly_d = np.array([[400, 200], [460, 200], [460, 260], [400, 260]], np.float32)
    class Boxes:
        def __init__(self, cls, conf):
            self.cls, self.conf = T(cls), T(conf)

        def __len__(self):
            return len(self.cls.a)

    res = SimpleNamespace(masks=SimpleNamespace(xy=[poly_y, poly_d]), boxes=Boxes([YELLOW, DUCKIE], [0.9, 0.8]))
    seg = YoloSegmenter.__new__(YoloSegmenter)
    seg.work_scale = 0.5
    out = seg.from_result(res, (640, 480))
    assert out.work_size == (320, 240)
    assert out.class_masks[YELLOW][175, 60] and not out.class_masks[YELLOW][100, 60]
    assert len(out.instances) == 1 and out.instances[0].cls == DUCKIE
