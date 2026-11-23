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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="small_loop")
    ap.add_argument("--weights", default="")
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
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--video", default="")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--raw-actuation", action="store_true", help="do not compensate the sim motor dynamics")
    ap.add_argument("--log-csv", default="", help="per-step log (gt vs estimate, command)")
    a = ap.parse_args(argv)

    agent, kind = build_agent(a)
    env = make_env(a.map, seed=a.seed, domain_rand=a.domain_rand, max_steps=a.steps, width=a.width, height=a.height)
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
