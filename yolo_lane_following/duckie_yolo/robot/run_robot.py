"""Drive the real Duckiebot with the YOLO lane-following agent (no ROS).

Talks to robot_io over ZMQ, so it runs either on the robot itself (``--host localhost``,
e.g. inside the Ultralytics JetPack container with a TensorRT engine) or on a laptop over
Wi-Fi (``--host <robot>.local --encoding jpeg``).

Examples:
    # on the Orin, after `python3 -m duckie_yolo.robot.robot_io` is running:
    python3 -m duckie_yolo.robot.run_robot --weights models/best.engine --robot-config configs/robot_mybot.yaml
    # perception only, wheels stay still, watch http://<robot>:8080
    python3 -m duckie_yolo.robot.run_robot --weights models/best.engine --dry-run
    # classical colour segmentation instead of YOLO (bring-up / debugging)
    python3 -m duckie_yolo.robot.run_robot --segmenter color --dry-run
"""
from __future__ import annotations

import argparse
import signal
import time

import numpy as np

from ..agent import LaneFollowingAgent
from ..config import load_agent_config
from ..perception.segmenters import build_segmenter
from ..viz import compose
from .real_env import RealRobotEnv
from .robot_config import RobotConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--encoding", default="raw", choices=["raw", "jpeg"], help="use jpeg over Wi-Fi")
    ap.add_argument("--weights", default="", help=".pt / .onnx / .engine (TensorRT on the Orin)")
    ap.add_argument("--segmenter", default=None, choices=["yolo", "color"])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", default=None, help="e.g. 0 / cuda:0 / cpu / mps")
    ap.add_argument("--agent-config", default="", help="YAML overriding duckie_yolo.config.AgentConfig")
    ap.add_argument("--robot-config", default="", help="calibration YAML (see calibrate_*.py)")
    ap.add_argument("--max-speed", type=float, default=0.3, help="cap on the forward speed command")
    ap.add_argument("--speed", type=float, default=None, help="override controller.v_ref")
    ap.add_argument("--no-stoplines", action="store_true")
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="never move the wheels")
    ap.add_argument("--stream-port", type=int, default=8080, help="MJPEG debug stream (0 = off)")
    ap.add_argument("--record", default="", help="save the debug view to this .mp4")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl-C)")
    a = ap.parse_args(argv)

    cfg = load_agent_config(a.agent_config)
    if a.weights:
        cfg.segmenter.weights = a.weights
    if a.imgsz:
        cfg.segmenter.imgsz = a.imgsz
    if a.conf is not None:
        cfg.segmenter.conf = a.conf
    if a.device is not None:
        cfg.segmenter.device = a.device
    if a.speed is not None:
        cfg.controller.v_ref = a.speed
    cfg.behavior.enable_stoplines = not a.no_stoplines
    cfg.behavior.enable_obstacles = not a.no_obstacles

    robot = RobotConfig.load(a.robot_config)
    if not all(robot.calibrated.values()):
        print(f"[run_robot] WARNING: robot not fully calibrated {robot.calibrated} - using defaults where missing")
    camera = robot.camera_model()
    if cfg.actuation == type(cfg.actuation)():
        cfg.actuation = robot.actuation  # from calibrate_wheels (--distance / --rotate)
    segmenter = build_segmenter(cfg.segmenter, cfg.estimator.work_scale, a.segmenter)
    agent = LaneFollowingAgent(segmenter, camera, cfg)
    env = RealRobotEnv(a.host, a.port, robot.wheels, encoding=a.encoding)

    stream = None
    if a.stream_port:
        from .mjpeg import MJPEGServer

        stream = MJPEGServer(a.stream_port)
        print(f"[run_robot] debug stream on http://<this-host>:{a.stream_port}/")
    writer = None

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))

    obs = env.reset()
    # warm-up inference (TensorRT/CUDA init) before the wheels can move
    agent.act(obs)
    agent.reset()
    t0 = time.monotonic()
    n, fps, t_last = 0, 0.0, time.monotonic()
    action = np.zeros(2)
    try:
        while not stop["now"]:
            step = agent.act(obs)
            action = step.action.copy()
            action[0] = min(action[0], a.max_speed)
            if a.dry_run:
                action[:] = 0.0
            obs, _, _, info = env.step(action)
            n += 1
            now = time.monotonic()
            if now - t_last >= 1.0:
                fps = n / (now - t_last)
                n, t_last = 0, now
            if stream is not None or a.record:
                e = step.estimate
                lines = [
                    f"{step.state.value}{' (DRY RUN)' if a.dry_run else ''}  v={action[0]:.2f} w={action[1]:+.2f}",
                    f"d={e.d:+.3f} phi={np.degrees(e.phi):+.1f}deg conf={e.confidence:.2f} src={e.source}",
                    f"{fps:.1f} Hz  seg {1000 * step.timings['segment']:.0f} ms  total {1000 * step.timings['total']:.0f} ms",
                ]
                frame = compose(step.image, step.segmentation, e, camera, lines)
                if stream is not None:
                    stream.publish(frame)
                if a.record:
                    import cv2

                    if writer is None:
                        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
            if a.duration and now - t0 > a.duration:
                break
    finally:
        env.close()
        if writer is not None:
            writer.release()
        if stream is not None:
            stream.close()
        print("[run_robot] stopped, wheels released")


if __name__ == "__main__":
    main()
