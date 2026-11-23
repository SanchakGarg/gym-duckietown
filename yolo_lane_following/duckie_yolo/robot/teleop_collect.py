"""Keyboard teleoperation from a laptop + recording of rectified frames (for real-world fine-tuning).

    python -m duckie_yolo.robot.teleop_collect --host <robot>.local --robot-config configs/robot_mybot.yaml --out data/real/raw

Keys (OpenCV window must have focus):  w/s set forward/backward cruise speed, a/d turn while
held, space stop, r toggle recording, q / Esc quit. If this program dies, robot_io's
watchdog releases the wheels.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from .real_env import RealRobotEnv
from .robot_config import RobotConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--robot-config", default="")
    ap.add_argument("--out", default="data/real/raw")
    ap.add_argument("--hz", type=float, default=3.0, help="recording rate")
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--turn", type=float, default=2.0)
    a = ap.parse_args(argv)

    robot = RobotConfig.load(a.robot_config)
    cam = robot.camera_model()
    env = RealRobotEnv(a.host, a.port, robot.wheels, encoding="jpeg", jpeg_quality=90)
    os.makedirs(a.out, exist_ok=True)
    obs = env.reset()
    v = w = 0.0
    recording, n_saved, t_last = False, 0, 0.0
    try:
        while True:
            rect = cv2.cvtColor(cam.rectify(obs), cv2.COLOR_RGB2BGR)
            vis = rect.copy()
            cv2.putText(vis, f"{'REC' if recording else 'rec off'} saved={n_saved} v={v:.2f} w={w:+.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (255, 255, 255), 2)
            cv2.imshow("teleop", vis)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            # w/s set a cruise speed that is kept until space; a/d turn only while held
            if k == ord("w"):
                v = a.speed
            elif k == ord("s"):
                v = -a.speed
            elif k == ord(" "):
                v = 0.0
            w = a.turn if k == ord("a") else -a.turn if k == ord("d") else 0.0
            if k == ord("r"):
                recording = not recording
            if recording and time.time() - t_last >= 1.0 / a.hz:
                cv2.imwrite(os.path.join(a.out, f"real_{int(time.time() * 1000)}.jpg"), rect)
                n_saved += 1
                t_last = time.time()
            obs, *_ = env.step([v, w])
    finally:
        env.close()
        cv2.destroyAllWindows()
        print(f"[teleop] saved {n_saved} frames to {a.out}")


if __name__ == "__main__":
    main()
