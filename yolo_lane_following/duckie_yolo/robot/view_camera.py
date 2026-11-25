"""Live camera viewer for robot_io: run it on a laptop. The wheels never move.

    python -m duckie_yolo.robot.view_camera --host <robot>.local
    python -m duckie_yolo.robot.view_camera --host <robot>.local --robot-config configs/robot_mybot.yaml  # + rectified

Every request is a zero-duty "action", so robot_io keeps the wheels stopped. robot_io serves one
client at a time: close this before you start run_robot / teleop_collect / calibrate_*.
Keys (window focused): s save a snapshot, q / Esc quit.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .real_env import RealRobotEnv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--robot-config", default="", help="if given, also show the rectified (sim-like) view")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality over Wi-Fi")
    a = ap.parse_args(argv)

    cam = None
    if a.robot_config:
        from .robot_config import RobotConfig

        cam = RobotConfig.load(a.robot_config).camera_model()
    env = RealRobotEnv(a.host, a.port, encoding="jpeg", jpeg_quality=a.quality)
    print(f"[view_camera] connected to {env.addr}  (s = snapshot, q = quit)")
    n, fps, t_last = 0, 0.0, time.monotonic()
    try:
        while True:
            rgb = env.wheels(0.0, 0.0)
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if cam is not None:
                rect = cv2.cvtColor(cam.rectify(rgb), cv2.COLOR_RGB2BGR)
                rect = cv2.resize(rect, (int(rect.shape[1] * view.shape[0] / rect.shape[0]), view.shape[0]))
                view = np.hstack([view, rect])
            n += 1
            now = time.monotonic()
            if now - t_last >= 1.0:
                fps, n, t_last = n / (now - t_last), 0, now
            label = f"{fps:4.1f} fps  {rgb.shape[1]}x{rgb.shape[0]}" + ("   raw | rectified" if cam is not None else "")
            cv2.putText(view, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("duckiebot camera", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(name, view)
                print(f"[view_camera] saved {name}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
