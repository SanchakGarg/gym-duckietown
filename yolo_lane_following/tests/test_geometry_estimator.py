import math

import cv2
import numpy as np
import pytest

from duckie_yolo.classes import DUCKIE, RED, WHITE, YELLOW
from duckie_yolo.config import AgentConfig
from duckie_yolo.estimation.lane_estimator import LaneEstimator, ransac_polyfit
from duckie_yolo.perception.camera_model import CameraModel
from duckie_yolo.perception.segmenters import segmentation_from_label_map

CAM = CameraModel.sim_default()
CFG = AgentConfig()


def test_homography_roundtrip():
    xy = np.array([[0.2, 0.0], [0.5, 0.1], [0.3, -0.2], [0.8, 0.3]])
    uv, ok = CAM.ground_to_image(xy)
    assert ok.all()
    back, ok2 = CAM.image_to_ground(uv)
    assert ok2.all()
    np.testing.assert_allclose(back, xy, atol=1e-9)


def test_sim_camera_geometry():
    # straight ahead projects onto the centre column, further away is higher in the image
    uv, _ = CAM.ground_to_image(np.array([[0.3, 0.0], [0.6, 0.0], [0.3, 0.1]]))
    assert abs(uv[0, 0] - 320) < 1e-6 and abs(uv[1, 0] - 320) < 1e-6
    assert uv[1, 1] < uv[0, 1]
    assert uv[2, 0] < 320  # y > 0 (left) appears on the left
    # nearest visible floor: bottom row ~= 0.066 + 0.108 / tan(19.15 + 37.5 deg)
    p, ok = CAM.image_to_ground(np.array([[320, 480]]))
    assert ok[0] and abs(p[0, 0] - (0.066 + 0.108 / math.tan(math.radians(56.65)))) < 1e-3
    # rows above the horizon do not hit the floor
    assert 0 < CAM.horizon_row() < 240
    _, ok = CAM.image_to_ground(np.array([[320, CAM.horizon_row() - 5]]))
    assert not ok[0]


def render_lane(d: float, phi: float, curvature: float = 0.0, yellow=True, white=True, red_at=None, duckie_at=None):
    """Synthetic label map of a lane seen by the sim camera with the robot at (d, phi).

    Lane frame: s along the lane, n to the left. Robot at s=0, n=-d (d>0: right of center),
    heading rotated by -phi (phi>0: pointing right of the lane direction).
    """
    lm = np.full((480, 640), -1, np.int16)
    L = CFG.lane

    def to_robot(s, n):
        # point in lane frame (with constant curvature bending the lane to the left)
        if abs(curvature) > 1e-9:
            r = 1.0 / curvature
            th = s / r
            x_l, y_l = (r - n) * np.sin(th), r - (r - n) * np.cos(th)
        else:
            x_l, y_l = s, n
        x_l, y_l = x_l, y_l + d  # robot lateral position: center line at +d in the robot frame
        c, si = math.cos(phi), math.sin(phi)
        return np.column_stack([c * x_l - si * y_l, si * x_l + c * y_l])

    def poly_strip(n_center, width, s0=0.15, s1=1.2):
        s = np.linspace(s0, s1, 80)
        left = to_robot(s, np.full_like(s, n_center + width / 2))
        right = to_robot(s, np.full_like(s, n_center - width / 2))
        pts = np.vstack([left, right[::-1]])
        uv, ok = CAM.ground_to_image(pts)
        if ok.all():
            return np.round(uv).astype(np.int32)
        return None

    if yellow:
        for s0 in np.arange(0.15, 1.2, 0.10):  # dashes 5 cm long every 10 cm
            p = poly_strip(L.yellow_offset, 0.024, s0, s0 + 0.05)
            if p is not None:
                m = np.zeros_like(lm, np.uint8)
                cv2.fillPoly(m, [p], 1)
                lm[m.astype(bool)] = YELLOW
    if white:
        p = poly_strip(-L.white_offset, 0.048)
        m = np.zeros_like(lm, np.uint8)
        cv2.fillPoly(m, [p], 1)
        lm[m.astype(bool)] = WHITE
    if red_at is not None:
        s = red_at
        pts = to_robot(np.array([s, s, s + 0.05, s + 0.05]), np.array([-0.15, 0.10, 0.10, -0.15]))
        uv, _ = CAM.ground_to_image(pts)
        m = np.zeros_like(lm, np.uint8)
        cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
        lm[m.astype(bool)] = RED
    if duckie_at is not None:
        uv, _ = CAM.ground_to_image(to_robot(np.array([duckie_at]), np.array([0.0])))
        u, v = uv[0]
        cv2.ellipse(lm, (int(u), int(v) - 25), (18, 25), 0, 0, 360, int(DUCKIE), -1)
    return lm


@pytest.mark.parametrize("d,phi", [(0.0, 0.0), (0.05, 0.0), (-0.06, 0.0), (0.0, 0.2), (0.04, -0.25), (-0.03, 0.15)])
def test_estimator_recovers_pose_straight(d, phi):
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    seg = segmentation_from_label_map(render_lane(d, phi), CFG.estimator.work_scale)
    e = est.estimate(seg)
    assert e.valid and e.source == "both"
    assert abs(e.d - d) < 0.015, (e.d, d)
    assert abs(e.phi - phi) < math.radians(4), (math.degrees(e.phi), math.degrees(phi))


@pytest.mark.parametrize("which", ["yellow", "white"])
def test_estimator_single_line(which):
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    lm = render_lane(0.03, 0.1, yellow=which == "yellow", white=which == "white")
    e = est.estimate(segmentation_from_label_map(lm, CFG.estimator.work_scale))
    assert e.valid and e.source == which
    assert abs(e.d - 0.03) < 0.02 and abs(e.phi - 0.1) < math.radians(5)


def test_estimator_curve_lookahead_turns_left():
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    e = est.estimate(segmentation_from_label_map(render_lane(0.0, 0.0, curvature=1.5), CFG.estimator.work_scale))
    assert e.valid
    assert e.lookahead[1] > 0.02  # lane bends left -> target on the left
    assert abs(e.d) < 0.03


def test_ignores_opposite_lane_white_line():
    lm = render_lane(0.0, 0.0)
    # add the far white edge of the opposite lane (left of the yellow line)
    extra = render_lane(-(CFG.lane.yellow_offset * 2 + CFG.lane.white_offset), 0.0, yellow=False)
    lm[(extra == WHITE) & (lm < 0)] = WHITE
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    e = est.estimate(segmentation_from_label_map(lm, CFG.estimator.work_scale))
    assert e.valid and abs(e.d) < 0.02


def test_no_markings_is_invalid():
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    e = est.estimate(segmentation_from_label_map(np.full((480, 640), -1, np.int16), CFG.estimator.work_scale))
    assert not e.valid and e.lookahead is None


def test_stopline_and_obstacle_detection():
    est = LaneEstimator(CAM, CFG.estimator, CFG.lane)
    e = est.estimate(segmentation_from_label_map(render_lane(0.0, 0.0, red_at=0.3, duckie_at=0.35), CFG.estimator.work_scale))
    assert e.stopline_x is not None and abs(e.stopline_x - 0.3) < 0.03
    assert len(e.obstacles) == 1
    cls, x, y = e.obstacles[0]
    assert cls == DUCKIE and abs(x - 0.35) < 0.05 and abs(y) < 0.05


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 0.7, 200)
    y = 0.3 * x * x - 0.1 * x + 0.05 + rng.normal(0, 0.003, 200)
    y[:40] = rng.uniform(-0.5, 0.5, 40)
    fit = ransac_polyfit(x, y, 2, 50, 0.02, rng)
    np.testing.assert_allclose(fit.coeffs, [0.3, -0.1, 0.05], atol=0.03)
