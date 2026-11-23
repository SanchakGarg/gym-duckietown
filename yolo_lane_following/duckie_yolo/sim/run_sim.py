"""Run the lane-following agent in gym-duckietown (live window and/or video).

    python -m duckie_yolo.sim.run_sim --map small_loop --weights runs/segment/sim_v1/weights/best.pt
    python -m duckie_yolo.sim.run_sim --map loop_obstacles --oracle --video out.mp4 --no-show   # perfect masks
    python -m duckie_yolo.sim.run_sim --map 4way --segmenter color                             # HSV baseline
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..agent import LaneFollowingAgent
from ..config import ActuationConfig, load_agent_config
from ..perception.camera_model import CameraModel
from ..perception.segmenters import build_segmenter, segmentation_from_label_map
from ..viz import compose
from .labels import env_mesh_names, object_palette, render_label_map
from .sim_utils import SIM_OMEGA_RATIO, SIM_V_RATIO, gt_lane_pose, make_env, sim_time


def build_agent(args, cfg=None):
    cfg = cfg or load_agent_config(getattr(args, "agent_config", ""))
    if getattr(args, "weights", ""):
        cfg.segmenter.weights = args.weights
    if getattr(args, "speed", None):
        cfg.controller.v_ref = args.speed
    if getattr(args, "device", None):
        cfg.segmenter.device = args.device
    if getattr(args, "no_stoplines", False):
        cfg.behavior.enable_stoplines = False
    if getattr(args, "no_obstacles", False):
        cfg.behavior.enable_obstacles = False
    if not getattr(args, "raw_actuation", False) and cfg.actuation == ActuationConfig():
        # compensate the simulator's motor dynamics so (v, omega) mean physical units
        cfg.actuation.v_gain = 1.0 / SIM_V_RATIO
        cfg.actuation.omega_gain = 1.0 / SIM_OMEGA_RATIO
    kind = "oracle" if args.oracle else (args.segmenter or ("yolo" if cfg.segmenter.weights else "color"))
    seg = None if kind == "oracle" else build_segmenter(cfg.segmenter, cfg.estimator.work_scale, kind)
    camera = CameraModel.sim_default(args.width, args.height)
    return LaneFollowingAgent(seg, camera, cfg), kind


def top_down_view(env, follow: bool = True, half_width_m: float = 0.9, size: int = 480, lookahead=None) -> np.ndarray:
    """Bird's-eye render of the map *with the robot* (BGR), for humans only.

    Overlays: cyan = lane-centre paths of every lane (the middle of each lane, between the
    yellow and white lines - this is what "following the lane" means); green = the path of
    the robot's own lane ahead; red dot = the agent's current lookahead target; arrow = heading.
    ``follow`` crops a window of +-half_width_m around the robot.
    """
    from gym_duckietown.graphics import bezier_point
    from gym_duckietown.simulator import WINDOW_HEIGHT, WINDOW_WIDTH, get_dir_vec, get_right_vec

    u = env.unwrapped
    img = u._render_img(WINDOW_WIDTH, WINDOW_HEIGHT, u.multi_fbo_human, u.final_fbo_human, u.img_array_human,
                        top_down=True)
    img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    # the top-down camera looks straight down at the map centre (Simulator._render_img)
    ts = u.road_tile_size
    a, b = u.grid_width * ts / 2.0, u.grid_height * ts / 2.0
    ppm = (WINDOW_HEIGHT / 2.0) / (max(a, b) + 0.1)

    def px(p):
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        return np.column_stack([WINDOW_WIDTH / 2.0 + (p[:, 0] - a) * ppm, WINDOW_HEIGHT / 2.0 + (p[:, 2] - b) * ppm])

    lw = max(1, int(round(0.012 * ppm)))
    ts_ = np.linspace(0, 1, 16)
    for tile in u.drivable_tiles:
        for cps in tile["curves"]:
            pts = np.array([bezier_point(cps, t) for t in ts_])
            cv2.polylines(img, [px(pts).astype(np.int32)], False, (255, 200, 0), lw, cv2.LINE_AA)
    # own lane ahead: walk along the closest curve in the heading direction
    pos, ang = u.cur_pos.copy(), u.cur_angle
    path = []
    for _ in range(40):
        pt, tan = u.closest_curve_point(pos, ang)
        if pt is None:
            break
        path.append(pt)
        tan = tan / max(np.linalg.norm(tan), 1e-9)
        pos = pt + 0.04 * tan
        ang = float(np.arctan2(-tan[2], tan[0]))
    if len(path) > 1:
        cv2.polylines(img, [px(np.array(path)).astype(np.int32)], False, (0, 220, 0), lw + 2, cv2.LINE_AA)
    d = get_dir_vec(u.cur_angle)
    left = -get_right_vec(u.cur_angle)  # robot frame y (left) on the floor
    c = px(u.cur_pos)[0]
    tip = px(u.cur_pos + 0.15 * d)[0]
    cv2.arrowedLine(img, tuple(int(v) for v in c), tuple(int(v) for v in tip), (255, 255, 255), lw + 1, cv2.LINE_AA, tipLength=0.3)
    if lookahead is not None:
        target = u.cur_pos + lookahead[0] * d + lookahead[1] * left
        cv2.circle(img, tuple(int(v) for v in px(target)[0]), lw + 4, (0, 0, 255), -1, cv2.LINE_AA)
    if not follow:
        return cv2.resize(img, (int(size * WINDOW_WIDTH / WINDOW_HEIGHT), size))
    r = max(8, int(half_width_m * ppm))
    pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    crop = pad[int(c[1]) : int(c[1]) + 2 * r, int(c[0]) : int(c[0]) + 2 * r]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="small_loop")
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default=None, help="YOLO inference device: cpu | mps | 0")
    ap.add_argument("--segmenter", default=None, choices=["yolo", "color"])
    ap.add_argument("--oracle", action="store_true", help="use ground-truth masks from the segmentation render")
    ap.add_argument("--agent-config", default="")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--no-stoplines", action="store_true")
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--domain-rand", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-angle", type=float, default=45.0,
                    help="max heading error (deg) of the random start pose (simulator default 45)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--video", default="")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--raw-actuation", action="store_true", help="do not compensate the sim motor dynamics")
    ap.add_argument("--log-csv", default="", help="per-step log (gt vs estimate, command)")
    ap.add_argument("--top-down", nargs="?", const="follow", default="", choices=["follow", "full"],
                    help="also show a bird's-eye view with the robot (follow: zoomed on the robot, full: whole map)")
    a = ap.parse_args(argv)

    agent, kind = build_agent(a)
    env = make_env(a.map, seed=a.seed, domain_rand=a.domain_rand, max_steps=a.steps, width=a.width, height=a.height,
                   accept_start_angle_deg=a.start_angle)
    palette = object_palette(env_mesh_names(env))
    writer = None
    log = open(a.log_csv, "w") if a.log_csv else None
    if log:
        log.write("ep,step,gt_d,gt_phi,est_d,est_phi,valid,src,conf,look_x,look_y,v,w,state\n")
    for ep in range(a.episodes):
        env.unwrapped.seed(a.seed + ep)
        obs = env.reset()
        agent.reset()
        errs = []
        for step in range(a.steps):
            seg = None
            if kind == "oracle":
                seg = segmentation_from_label_map(render_label_map(env, palette), agent.cfg.estimator.work_scale)
            res = agent.act(obs, t=sim_time(env), segmentation=seg)
            gt = gt_lane_pose(env)
            e = res.estimate
            if gt is not None and e.valid:
                errs.append((e.d - gt[0], e.phi - gt[1]))
            if log:
                lx, ly = e.lookahead if e.lookahead else (float('nan'), float('nan'))
                g = gt if gt else (float('nan'), float('nan'))
                log.write(f"{ep},{step},{g[0]:.4f},{g[1]:.4f},{e.d:.4f},{e.phi:.4f},{int(e.valid)},{e.source},{e.confidence:.2f},"
                          f"{lx:.3f},{ly:.3f},{res.action[0]:.3f},{res.action[1]:.3f},{res.state.value}\n")
            if not a.no_show or a.video:
                lines = [
                    f"{a.map} [{kind}] step {step} {res.state.value}  v={res.action[0]:.2f} w={res.action[1]:+.2f}",
                    f"est d={e.d:+.3f} phi={np.degrees(e.phi):+.1f}  conf={e.confidence:.2f} {e.source}",
                    f"gt  d={gt[0]:+.3f} phi={np.degrees(gt[1]):+.1f}" if gt else "gt: not in lane",
                ]
                frame = compose(res.image, res.segmentation, e, agent.camera, lines)
                if a.top_down:
                    td = top_down_view(env, follow=a.top_down == "follow", size=frame.shape[0], lookahead=e.lookahead)
                    for i, (txt, col) in enumerate([("green: lane centre it follows", (0, 220, 0)),
                                                    ("cyan: all lane centres", (255, 200, 0)),
                                                    ("red: steering target", (0, 0, 255))]):
                        cv2.putText(td, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(td, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                    frame = np.hstack([frame, td])
                if a.video:
                    if writer is None:
                        writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
                if not a.no_show:
                    cv2.imshow("duckie_yolo", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            obs, _r, done, info = env.step(res.action)
            if done:
                print(f"[run_sim] episode {ep} ended at step {step}: {info.get('Simulator', {}).get('msg', '')}")
                break
        if errs:
            er = np.abs(np.array(errs))
            print(f"[run_sim] episode {ep}: {step + 1} steps, estimator |d| err {er[:, 0].mean():.3f} m, |phi| err {np.degrees(er[:, 1].mean()):.1f} deg")
    if writer is not None:
        writer.release()
    if log:
        log.close()
    env.close()


if __name__ == "__main__":
    main()
