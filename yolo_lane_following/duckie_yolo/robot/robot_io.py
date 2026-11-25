"""robot_io: the only process that touches the Duckiebot hardware. No ROS, no Duckietown stack.

Runs natively on the robot (Jetson Orin Nano / Nano / Raspberry Pi). It owns the camera and
the wheel motors and serves them over ZMQ (see protocol.py) to the agent, which can run on
the robot itself (localhost, e.g. inside the Ultralytics JetPack container) or on a laptop.

Closed loop (robot/drive.py): every wheel command passes through a 100 Hz drive controller with
per-wheel start power, a start kick and, with the IMU gyro, a yaw-rate PID. Clients set it with
the "configure" command; "drive_task" runs the IMU / start-power self-tests.

Safety:
  * watchdog: wheels are released if no command arrives for --watchdog seconds
  * --max-duty caps the wheel duty cycle
  * SIGINT/SIGTERM release the wheels and the GPIO pins

Examples (on the robot):
    python3 -m duckie_yolo.robot.robot_io                        # serve camera + wheels on :5555
    python3 -m duckie_yolo.robot.robot_io --hw-test              # spin each wheel, save a frame, exit
    python3 -m duckie_yolo.robot.robot_io --camera mock --mock-wheels   # no hardware (testing)
"""
from __future__ import annotations

import argparse
import json
import signal
import time

from .hw.camera import open_camera
from .protocol import DEFAULT_PORT, encode_frame


def build_wheels(args):
    from .hw.motors import WheelPWMConfig, Wheels, fake_wheels

    pwm = WheelPWMConfig(args.pwm_min, args.pwm_max, args.deadzone)
    kw = dict(pwm_cfg=pwm, invert_left=args.invert_left, invert_right=args.invert_right, max_duty=args.max_duty)
    if args.mock_wheels:
        return fake_wheels(args.hat, **kw)[0]
    return Wheels.open(args.i2c_bus, args.i2c_addr, args.hat, **kw)


def build_drive(args, wheels):
    """Drive controller with the IMU gyro if one answers on the HAT's I2C bus."""
    from .drive import DriveController

    imu = None
    if not getattr(args, "mock_wheels", False) and not getattr(args, "no_imu", False):
        try:
            from .hw.imu import GyroZ

            imu = GyroZ.open(args.i2c_bus, args.imu_addr)
            print(f"[robot_io] IMU {imu.chip} at 0x{args.imu_addr:02x}: yaw-rate PID available")
        except Exception as e:
            print(f"[robot_io] no IMU ({e}): start power + kick only, no PID")
    return DriveController(wheels, imu)


def hw_test(args, wheels, cam) -> None:
    import cv2

    print("[hw-test] lift the robot so the wheels spin freely!")
    time.sleep(2.0)
    for name, (l, r) in [("left fwd", (0.5, 0)), ("left back", (-0.5, 0)), ("right fwd", (0, 0.5)), ("right back", (0, -0.5)),
                         ("both fwd", (0.5, 0.5))]:
        print(f"[hw-test] {name}")
        wheels.set(l, r)
        time.sleep(1.0)
        wheels.stop()
        time.sleep(0.4)
    frame, stamp, seq = cam.latest(timeout=3.0)
    if frame is None:
        print(f"[hw-test] NO CAMERA FRAME (error: {cam.error})")
    else:
        cv2.imwrite(args.hw_test_image, frame)
        print(f"[hw-test] saved camera frame {frame.shape} -> {args.hw_test_image}")


def serve(args, wheels, cam, drive=None) -> None:
    import zmq

    from .drive import DriveController

    if drive is None:
        drive = DriveController(wheels)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    # detect dead peers so a crashed client can reconnect to the PAIR socket
    sock.setsockopt(zmq.HEARTBEAT_IVL, 500)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, 2000)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[robot_io] serving on tcp://{args.bind}:{args.port}  (watchdog {args.watchdog}s, max duty {args.max_duty})")

    running = {"on": True}

    def _stop(signum, _frame):
        running["on"] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_cmd = 0.0
    wheels_active = False
    n_req = 0
    t_report = time.time()
    while running["on"]:
        events = dict(poller.poll(50))
        now = time.time()
        if sock in events:
            try:
                msg = sock.recv_json(zmq.NOBLOCK)
            except (zmq.Again, json.JSONDecodeError, ValueError):
                continue
            cmd = msg.get("command", "")
            if cmd == "action":
                vals = msg.get("values", [0.0, 0.0])
                wref = msg.get("omega_ref")
                drive.command(float(vals[0]), float(vals[1]), None if wref is None else float(wref))
                wheels_active = True
                last_cmd = now
            elif cmd in ("reset", "stop"):
                drive.stop()
                wheels_active = False
                last_cmd = now
            elif cmd == "ping":
                sock.send_json({"pong": True, "req_id": msg.get("req_id")})
                continue
            elif cmd == "configure":
                try:
                    t = drive.configure(msg.get("tuning") or {})
                    sock.send_json({"ok": True, "tuning": t.to_dict(), "drive": drive.status(), "req_id": msg.get("req_id")})
                except Exception as e:
                    sock.send_json({"ok": False, "error": str(e), "req_id": msg.get("req_id")})
                continue
            elif cmd == "drive_task":
                try:
                    task = drive.start_task(str(msg.get("task", "")))
                    sock.send_json({"ok": True, "task": task, "req_id": msg.get("req_id")})
                except Exception as e:
                    sock.send_json({"ok": False, "error": str(e), "req_id": msg.get("req_id")})
                continue
            if cmd in ("reset", "action"):
                frame, stamp, seq = cam.latest(after_seq=int(msg.get("after_seq", -1)), timeout=args.frame_timeout)
                if frame is None:
                    sock.send_json({"error": f"no camera frame ({cam.error})", "req_id": msg.get("req_id")})
                    continue
                md, payload = encode_frame(frame, msg.get("encoding", "raw"), int(msg.get("quality", 80)))
                md.update(seq=seq, stamp=stamp, req_id=msg.get("req_id"), wheels=list(wheels.last), drive=drive.status())
                sock.send_json(md, zmq.SNDMORE)
                sock.send(payload, copy=False)
                n_req += 1
        if wheels_active and now - last_cmd > args.watchdog:
            drive.stop()
            wheels_active = False
            print("[robot_io] watchdog: no command received, wheels released")
        if now - t_report > 10.0:
            st = drive.status()
            print(f"[robot_io] {n_req / (now - t_report):.1f} req/s, wheels={wheels.last}, yaw rate={st['wz']:+.2f} rad/s, "
                  f"cam_err={cam.error}")
            n_req, t_report = 0, now
    print("[robot_io] shutting down")
    sock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="*")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--camera", default="csi", help="csi | usb:N | file:PATH | mock")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip-method", type=int, default=0, help="nvvidconv flip-method (2 = rotate 180)")
    ap.add_argument("--hat", default="v3", choices=["v1", "v2", "v3"], help="v3 for DB21/DB26 (Jetson)")
    ap.add_argument("--i2c-bus", type=int, default=7, help="7 on Orin Nano (DB26J), 1 on Jetson Nano")
    ap.add_argument("--i2c-addr", type=lambda s: int(s, 0), default=0x60)
    ap.add_argument("--pwm-min", type=int, default=60)
    ap.add_argument("--pwm-max", type=int, default=255)
    ap.add_argument("--deadzone", type=float, default=0.01)
    ap.add_argument("--invert-left", action="store_true")
    ap.add_argument("--invert-right", action="store_true")
    ap.add_argument("--max-duty", type=float, default=0.8)
    ap.add_argument("--watchdog", type=float, default=0.5)
    ap.add_argument("--frame-timeout", type=float, default=0.15)
    ap.add_argument("--mock-wheels", action="store_true", help="do not touch I2C/GPIO")
    ap.add_argument("--no-imu", action="store_true", help="do not use the IMU gyro (no yaw-rate PID)")
    ap.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x68)
    ap.add_argument("--hw-test", action="store_true")
    ap.add_argument("--hw-test-image", default="hw_test_frame.jpg")
    args = ap.parse_args(argv)

    cam = open_camera(args.camera, args.width, args.height, args.fps, args.flip_method)
    wheels = build_wheels(args)
    try:
        if args.hw_test:
            hw_test(args, wheels, cam)
        else:
            drive = build_drive(args, wheels)
            try:
                serve(args, wheels, cam, drive)
            finally:
                drive.close()
    finally:
        wheels.close()
        cam.stop()


if __name__ == "__main__":
    main()
