"""Keyboard teleoperation from a laptop + recording of rectified frames (for real-world fine-tuning).

    python -m duckie_yolo.robot.teleop_collect --host <robot>.local --robot-config configs/robot_mybot.yaml --out data/real/raw

Keys work in the terminal you started this from AND in the OpenCV window (whichever has focus;
the terminal is the reliable one under WSLg / remote X):
    w / s   set forward / backward cruise speed (kept until space)
    a / d   turn while held (a single tap turns for --turn-hold seconds)
    space   stop          r   toggle recording          q / Esc   quit
If this program dies, robot_io's watchdog releases the wheels.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import cv2

from .real_env import RealRobotEnv
from .robot_config import RobotConfig


class TerminalKeys:
    """Single key presses from the controlling terminal (cbreak mode: no Enter, no echo).

    Ctrl-C still raises KeyboardInterrupt; the terminal mode is restored by close().
    """

    def __init__(self):
        import termios
        import tty

        self._termios = termios
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._q: "queue.Queue[int]" = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        import select

        while not self._stop.is_set():
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if r:
                data = os.read(self.fd, 32)
                if not data:
                    return
                for b in data:
                    self._q.put(b)

    def keys(self) -> list:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()
        self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self._old)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--robot-config", default="")
    ap.add_argument("--out", default="data/real/raw")
    ap.add_argument("--hz", type=float, default=3.0, help="recording rate")
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--turn", type=float, default=2.0)
    ap.add_argument("--turn-hold", type=float, default=0.6,
                    help="seconds a/d keep turning after the last key event (covers the key-repeat delay)")
    ap.add_argument("--no-terminal-keys", action="store_true", help="only read keys from the OpenCV window")
    ap.add_argument("--no-window", action="store_true", help="no camera window (terminal keys only)")
    a = ap.parse_args(argv)

    robot = RobotConfig.load(a.robot_config)
    cam = robot.camera_model()
    env = RealRobotEnv(a.host, a.port, robot.wheels, encoding="jpeg", jpeg_quality=90)
    os.makedirs(a.out, exist_ok=True)
    term = None
    if not a.no_terminal_keys and sys.stdin.isatty():
        term = TerminalKeys()
    elif a.no_window:
        raise SystemExit("--no-window needs terminal keys (run it from a terminal)")
    print("[teleop] keys: w/s cruise fwd/back, a/d turn, space stop, r record, q quit"
          + ("  (type them in this terminal or the window)" if term else "  (in the window)"), flush=True)
    obs = env.reset()
    v = w = 0.0
    turn_until = 0.0
    recording, n_saved, t_last = False, 0, 0.0
    try:
        while True:
            rect = cv2.cvtColor(cam.rectify(obs), cv2.COLOR_RGB2BGR)
            keys = term.keys() if term else []
            if not a.no_window:
                vis = rect.copy()
                cv2.putText(vis, f"{'REC' if recording else 'rec off'} saved={n_saved} v={v:.2f} w={w:+.1f}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (255, 255, 255), 2)
                cv2.imshow("teleop", vis)
                k = cv2.waitKey(30) & 0xFF
                if k != 0xFF:
                    keys.append(k)
            else:
                time.sleep(0.03)
            if any(k in (ord("q"), ord("Q"), 27) for k in keys):
                break
            now = time.time()
            for k in keys:
                k = ord(chr(k).lower()) if k < 128 else k
                if k == ord("w"):
                    v = a.speed
                elif k == ord("s"):
                    v = -a.speed
                elif k == ord(" "):
                    v, w, turn_until = 0.0, 0.0, 0.0
                elif k in (ord("a"), ord("d")):
                    w = a.turn if k == ord("a") else -a.turn
                    turn_until = now + a.turn_hold
                elif k == ord("r"):
                    recording = not recording
                    print(f"[teleop] recording {'ON' if recording else 'off'} ({n_saved} saved)", flush=True)
            if now > turn_until:
                w = 0.0
            if recording and now - t_last >= 1.0 / a.hz:
                cv2.imwrite(os.path.join(a.out, f"real_{int(now * 1000)}.jpg"), rect)
                n_saved += 1
                t_last = now
            obs, *_ = env.step([v, w])
    except KeyboardInterrupt:
        pass
    finally:
        if term is not None:
            term.close()
        env.close()
        if not a.no_window:
            cv2.destroyAllWindows()
        print(f"[teleop] saved {n_saved} frames to {a.out}")


if __name__ == "__main__":
    main()
