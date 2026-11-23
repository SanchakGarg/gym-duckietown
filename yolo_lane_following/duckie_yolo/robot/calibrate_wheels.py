"""Wheel calibration (trim / gain) by driving straight.

Put the robot on a straight line (e.g. tape on the floor), then:

    python -m duckie_yolo.robot.calibrate_wheels --host <robot> --robot-config configs/robot_mybot.yaml --v 0.25 --duration 3

The robot drives "straight" for --duration seconds with the current calibration. Measure:
  * lateral drift at the end (cm, + = ended up LEFT of the line)  -> --drift
  * distance travelled (m)                                        -> --distance
and re-run with them (no driving) to update the config:

    python -m duckie_yolo.robot.calibrate_wheels --robot-config ... --v 0.25 --duration 3 --drift 6 --distance 0.62 --save

Trim > 0 speeds up the right wheel (turns left); gain scales both wheels so that the
commanded speed matches the real speed.

Yaw-rate check (spin in place), then enter the measured angle to set actuation.omega_gain:

    python -m duckie_yolo.robot.calibrate_wheels --robot-config ... --rotate --omega 2.0 --duration 3
    python -m duckie_yolo.robot.calibrate_wheels --robot-config ... --rotate --omega 2.0 --duration 3 --turned-deg 290 --save
"""
from __future__ import annotations

import argparse
import math
import time

from .robot_config import RobotConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--robot-config", default="configs/robot_duckiebot.yaml")
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--drift", type=float, default=None, help="measured lateral drift (cm, + = left)")
    ap.add_argument("--distance", type=float, default=None, help="measured distance travelled (m)")
    ap.add_argument("--trim-gain", type=float, default=0.005, help="trim change per cm of drift per m travelled")
    ap.add_argument("--rotate", action="store_true", help="spin in place to calibrate the yaw-rate gain")
    ap.add_argument("--omega", type=float, default=2.0, help="commanded yaw rate for --rotate (rad/s)")
    ap.add_argument("--turned-deg", type=float, default=None, help="measured rotation after --rotate (deg)")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)

    robot = RobotConfig.load(a.robot_config)
    cal = robot.wheels
    if a.rotate:
        if a.turned_deg is None:
            from .real_env import RealRobotEnv

            print(f"[wheels] spinning in place at omega={a.omega} rad/s for {a.duration}s in 2 s (mark the heading!)")
            env = RealRobotEnv(a.host, a.port, cal, encoding="jpeg")
            env.reset()
            time.sleep(2.0)
            t0 = time.monotonic()
            try:
                while time.monotonic() - t0 < a.duration:
                    env.step([0.0, a.omega])
            finally:
                env.close()
            expected = math.degrees(a.omega * a.duration)
            print(f"[wheels] expected {expected:.0f} deg; re-run with --turned-deg <measured> --save")
            return
        ratio = math.radians(a.turned_deg) / (a.omega * a.duration)
        robot.actuation.omega_gain = 1.0 / max(ratio, 1e-3)
        print(f"[wheels] realized/commanded yaw = {ratio:.3f} -> actuation.omega_gain = {robot.actuation.omega_gain:.3f}")
        if a.save:
            robot.save(a.robot_config)
            print(f"[wheels] wrote {a.robot_config}")
        return
    if a.drift is None and a.distance is None:
        from .real_env import RealRobotEnv

        print(f"[wheels] gain={cal.gain:.3f} trim={cal.trim:+.3f}; driving v={a.v} for {a.duration}s in 2 s ...")
        env = RealRobotEnv(a.host, a.port, cal, encoding="jpeg")
        env.reset()
        time.sleep(2.0)
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < a.duration:
                env.step([a.v, 0.0])
        finally:
            env.close()
        print("[wheels] done. Measure drift (cm, + = left) and distance (m), then re-run with --drift/--distance --save")
        return

    if a.drift is not None:
        dist = a.distance or a.v * a.duration
        cal.trim -= a.trim_gain * a.drift / max(dist, 0.1)
    if a.distance is not None and a.distance > 0:
        cal.gain *= (a.v * a.duration) / a.distance
    print(f"[wheels] new gain={cal.gain:.3f} trim={cal.trim:+.4f}")
    if a.save:
        robot.wheels = cal
        robot.calibrated["wheels"] = True
        robot.save(a.robot_config)
        print(f"[wheels] wrote {a.robot_config}")


if __name__ == "__main__":
    main()
