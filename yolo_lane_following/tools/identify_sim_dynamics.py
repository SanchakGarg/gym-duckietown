"""Measure gym-duckietown's steady-state response to DuckietownEnv [v, omega] commands.

The daffy simulator integrates identified motor dynamics (duckietown-world), so the realized
speed / yaw rate differ from the nominal kinematics. Result on daffy (used by duckie_yolo.sim as
SIM_V_RATIO / SIM_OMEGA_RATIO): realized v = 0.699 v_cmd, realized omega = 0.445 omega_cmd,
~0.63 s to reach 90 % of the steady state.

    python tools/identify_sim_dynamics.py
"""
import numpy as np

from duckie_yolo.sim.sim_utils import make_env


def main():
    env = make_env("loop_empty", seed=0, domain_rand=False)
    u = env.unwrapped
    dt = u.delta_time
    print(" v_cmd  w_cmd |  v_real  w_real | v_ratio w_ratio | t90 (s)")
    for v in (0.2, 0.35):
        for w in (0.0, 0.5, 1.0, 2.0, 4.0):
            env.reset()
            ang, pos = [], []
            for _ in range(90):
                env.step(np.array([v, w]))
                ang.append(u.cur_angle)
                pos.append(u.cur_pos.copy())
            ang, pos = np.unwrap(np.array(ang)), np.array(pos)
            wr = (ang[-1] - ang[-31]) / (30 * dt)
            vr = np.linalg.norm(np.diff(pos[-31:], axis=0), axis=1).sum() / (30 * dt)
            winst = np.diff(ang) / dt
            t90 = (np.argmax(np.abs(winst) >= 0.9 * abs(wr)) + 1) * dt if abs(wr) > 1e-3 else float("nan")
            print(f" {v:5.2f}  {w:5.2f} | {vr:7.3f} {wr:7.3f} | {vr / v:7.3f} {wr / w if w else float('nan'):7.3f} | {t90:.2f}")
    env.close()


if __name__ == "__main__":
    main()
