"""Closed-loop evaluation in gym-duckietown.

    python -m duckie_yolo.sim.eval_sim --agent yolo --weights runs/segment/sim_v1/weights/best.pt \
        --maps loop_empty small_loop zigzag_dists ETHZ_autolab_technical_track --episodes 5 --domain-rand
    python -m duckie_yolo.sim.eval_sim --agent expert ...      # privileged upper bound
    python -m duckie_yolo.sim.eval_sim --agent oracle ...      # perfect masks: tests geometry + control

Per episode: survived steps (max --steps), distance driven, |d| / |phi| w.r.t. the lane (ground
truth from Simulator.get_lane_pos2), estimator error, out-of-lane fraction, stops.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from ..perception.segmenters import segmentation_from_label_map
from .expert import PurePursuitExpert
from .labels import env_mesh_names, object_palette, render_label_map
from .run_sim import build_agent
from .sim_utils import gt_lane_pose, make_env, sim_time

FIELDS = ["agent", "map", "episode", "steps", "survived", "done_reason", "distance_m", "mean_abs_d", "p95_abs_d",
          "mean_abs_phi_deg", "est_abs_d_err", "est_abs_phi_err_deg", "out_of_lane_frac", "stop_events", "ms_per_step"]


def run_episode(env, agent, kind, palette, steps, expert_speed):
    u = env.unwrapped
    obs = env.reset()
    if agent is not None:
        agent.reset()
    expert = PurePursuitExpert(env, ref_velocity=expert_speed) if kind == "expert" else None
    ds, phis, est_err, out_lane, stops = [], [], [], 0, 0
    dist, last_pos, last_state = 0.0, u.cur_pos.copy(), None
    reason, t_agent = "max-steps", 0.0
    step = 0
    for step in range(steps):
        gt = gt_lane_pose(env)
        t0 = time.perf_counter()
        if expert is not None:
            action = np.array(expert.predict())
        else:
            seg = None
            if kind == "oracle":
                seg = segmentation_from_label_map(render_label_map(env, palette), agent.cfg.estimator.work_scale)
            res = agent.act(obs, t=sim_time(env), segmentation=seg)
            action = res.action
            if res.state.value != last_state and res.state.value in ("STOP_LINE_WAIT", "OBSTACLE_STOP"):
                stops += 1
            last_state = res.state.value
            if gt is not None and res.estimate.valid:
                est_err.append((abs(res.estimate.d - gt[0]), abs(res.estimate.phi - gt[1])))
        t_agent += time.perf_counter() - t0
        if gt is None:
            out_lane += 1
        else:
            ds.append(abs(gt[0]))
            phis.append(abs(gt[1]))
        obs, _r, done, info = env.step(action)
        dist += float(np.linalg.norm(u.cur_pos - last_pos))
        last_pos = u.cur_pos.copy()
        if done:
            msg = info.get("Simulator", {}).get("msg", "")
            reason = "invalid-pose" if "invalid pose" in msg else ("max-steps" if "max_steps" in msg else msg or "done")
            break
    n = step + 1
    ee = np.array(est_err) if est_err else np.full((1, 2), np.nan)
    return {
        "steps": n,
        "survived": int(reason == "max-steps"),
        "done_reason": reason,
        "distance_m": round(dist, 3),
        "mean_abs_d": round(float(np.mean(ds)), 4) if ds else np.nan,
        "p95_abs_d": round(float(np.percentile(ds, 95)), 4) if ds else np.nan,
        "mean_abs_phi_deg": round(float(np.degrees(np.mean(phis))), 2) if phis else np.nan,
        "est_abs_d_err": round(float(np.nanmean(ee[:, 0])), 4),
        "est_abs_phi_err_deg": round(float(np.degrees(np.nanmean(ee[:, 1]))), 2),
        "out_of_lane_frac": round(out_lane / n, 3),
        "stop_events": stops,
        "ms_per_step": round(1000 * t_agent / n, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="yolo", choices=["yolo", "color", "oracle", "expert"])
    ap.add_argument("--weights", default="")
    ap.add_argument("--agent-config", default="")
    ap.add_argument("--maps", nargs="+", default=["loop_empty", "small_loop", "zigzag_dists"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--domain-rand", action="store_true")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--expert-speed", type=float, default=0.35)
    ap.add_argument("--no-stoplines", action="store_true")
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default="runs/eval/eval.csv")
    ap.add_argument("--raw-actuation", action="store_true", help="do not compensate the sim motor dynamics")
    a = ap.parse_args(argv)

    a.oracle = a.agent == "oracle"
    a.segmenter = a.agent if a.agent in ("yolo", "color") else None
    agent, kind = (None, "expert") if a.agent == "expert" else build_agent(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = []
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in a.maps:
            env = make_env(m, seed=a.seed, domain_rand=a.domain_rand, max_steps=a.steps + 10, width=a.width, height=a.height,
                           accept_start_angle_deg=20)
            palette = object_palette(env_mesh_names(env))
            for ep in range(a.episodes):
                env.unwrapped.seed(a.seed + 1000 * ep)
                r = run_episode(env, agent, kind, palette, a.steps, a.expert_speed)
                r.update(agent=a.agent, map=m, episode=ep)
                rows.append(r)
                w.writerow(r)
                f.flush()
                print(f"[eval] {a.agent} {m} ep{ep}: {r['steps']} steps ({r['done_reason']}), {r['distance_m']} m, "
                      f"|d| {r['mean_abs_d']} m, est err {r['est_abs_d_err']} m, {r['ms_per_step']} ms/step", flush=True)
            env.close()
    print("\n[eval] summary per map")
    for m in a.maps:
        rr = [r for r in rows if r["map"] == m]
        print(f"  {m:32s} survived {sum(r['survived'] for r in rr)}/{len(rr)}  "
              f"dist {np.mean([r['distance_m'] for r in rr]):6.2f} m  "
              f"|d| {np.nanmean([r['mean_abs_d'] for r in rr]):.3f} m  "
              f"|phi| {np.nanmean([r['mean_abs_phi_deg'] for r in rr]):5.1f} deg  "
              f"est err {np.nanmean([r['est_abs_d_err'] for r in rr]):.3f} m")
    tot = sum(r["survived"] for r in rows)
    print(f"  TOTAL survived {tot}/{len(rows)} ({100 * tot / max(1, len(rows)):.0f}%)  -> {a.out}")


if __name__ == "__main__":
    main()
