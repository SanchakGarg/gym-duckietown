"""Duckiebot web interface: camera, models, teleop + dataset recording, calibration, guidelines.

Runs ON THE ROBOT next to robot_io; everything is used from a browser:

    python3 -m duckie_yolo.robot.webui                     # then open http://<robot-ip>:8080
    bash duckie_yolo/robot/webui/start_webui.sh            # same, inside the GPU Docker container (default)

Run it from yolo_lane_following/ (models/, data/real/, calib/ and configs/ are relative to it).
"""
from __future__ import annotations

import argparse
import os
import signal
import socket

from .calibration import Calibration
from .controller import RobotController
from .models import JobRunner, ModelManager, ultralytics_available
from .recorder import Recorder
from .server import WebUI, serve


def _ips() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<robot-ip>"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--robot-host", default="localhost", help="where robot_io runs")
    ap.add_argument("--robot-port", type=int, default=5555)
    ap.add_argument("--encoding", default="raw", choices=["raw", "jpeg"], help="jpeg if robot_io is on another machine")
    ap.add_argument("--robot-config", default="configs/robot_mybot.yaml", help="calibration file (created on first calibration)")
    ap.add_argument("--agent-config", default="configs/agent_real.yaml")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data-dir", default="data/real")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--stream-fps", type=float, default=15.0)
    a = ap.parse_args(argv)

    models = ModelManager(a.models_dir, a.agent_config)
    recorder = Recorder(a.data_dir)
    ctl = RobotController(a.robot_host, a.robot_port, a.robot_config, models, recorder, a.encoding, a.stream_fps).start()
    calib = Calibration(ctl, a.calib_dir)
    app = WebUI(ctl, calib, JobRunner(), workdir=os.getcwd())
    httpd = serve(app, a.bind, a.port)

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)  # `docker stop` -> release the wheels cleanly
    print(f"[webui] open http://{_ips()}:{a.port}/   (robot_io {a.robot_host}:{a.robot_port}, "
          f"YOLO {'available' if ultralytics_available() else 'NOT available - colour thresholds only'})", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        print("[webui] shutting down, wheels released", flush=True)
        ctl.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
