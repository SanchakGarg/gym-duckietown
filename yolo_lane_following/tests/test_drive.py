"""Drive controller (start power, kick, yaw-rate PID), IMU driver and the robot_io protocol for them."""
import argparse
import math
import socket
import threading
import time

import pytest

from duckie_yolo.robot.drive import DriveController, MotorTuning, apply_floor
from duckie_yolo.robot.hw.imu import FakeIMURegisters, GyroZ
from duckie_yolo.robot.hw.motors import fake_wheels


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += max(dt, 0.0)


class Plant:
    """Yaw response of a robot with mismatched motors and stiction, seen by a biased, flipped gyro."""

    def __init__(self, wheels, gain=8.0, left_eff=1.0, right_eff=0.8, stall=(0.0, 0.0), bias=0.05, sign=-1.0,
                 tau=0.0):
        self.wheels, self.gain = wheels, gain
        self.eff = (left_eff, right_eff)
        self.stall = stall
        self.bias, self.sign = bias, sign
        self.tau = tau  # motor/robot inertia (s); 0 = responds instantly
        self.state = 0.0
        self.chip = "FAKE"

    def target_wz(self):
        l, r = self.wheels.last
        l = 0.0 if abs(l) < self.stall[0] else l * self.eff[0]
        r = 0.0 if abs(r) < self.stall[1] else r * self.eff[1]
        return self.gain * (r - l)

    def advance(self, dt):
        if self.tau > 0:
            self.state += (self.target_wz() - self.state) * min(1.0, dt / self.tau)

    def true_wz(self):
        return self.state if self.tau > 0 else self.target_wz()

    def read_wz(self):
        return self.sign * self.true_wz() + self.bias


def make(tuning=None, **plant_kw):
    wheels, _, _ = fake_wheels("v3")
    plant = Plant(wheels, **plant_kw)
    ft = FakeTime()
    ctl = DriveController(wheels, plant, tuning, clock=ft.clock, sleep=ft.sleep, start_thread=False)
    return ctl, plant, wheels, ft


def run(ctl, ft, seconds, left, right, wref, tail=None):
    """Simulate `seconds` of driving; returns the integrated true heading change (rad), or with
    `tail` the mean true yaw rate over the last `tail` seconds."""
    heading, samples = 0.0, []
    ctl.command(left, right, wref)
    n = int(seconds / ctl.dt)
    for i in range(n):
        ft.sleep(ctl.dt)
        ctl.imu.advance(ctl.dt)
        with ctl._lock:
            ctl._apply_locked(ft.clock())
        heading += ctl.imu.true_wz() * ctl.dt
        if tail is not None and i >= n - int(tail / ctl.dt):
            samples.append(ctl.imu.true_wz())
    return sum(samples) / len(samples) if tail is not None else heading


def calibrated(**kw):
    t = MotorTuning(imu_calibrated=True, imu_sign=-1.0, gyro_bias=0.05, **kw)
    return t


def test_apply_floor():
    assert apply_floor(0.0, 0.3) == 0.0
    assert apply_floor(0.005, 0.3) == 0.0  # below the dead zone: off
    assert apply_floor(0.1, 0.3) == pytest.approx(0.3 + 0.7 * 0.1)
    assert apply_floor(-1.0, 0.3) == pytest.approx(-1.0)
    assert apply_floor(0.5, 0.0) == pytest.approx(0.5)


def test_pid_drives_straight_despite_weak_motor():
    open_loop, _, _, ft = make(calibrated(pid_enabled=False))
    drift_open = run(open_loop, ft, 3.0, 0.3, 0.3, 0.0)
    closed, _, _, ft2 = make(calibrated(pid_enabled=True, kp=0.06, ki=0.5), tau=0.1)
    drift_pid = run(closed, ft2, 3.0, 0.3, 0.3, 0.0)
    assert abs(math.degrees(drift_open)) > 30  # weak right motor: veers right by a lot
    assert abs(math.degrees(drift_pid)) < abs(math.degrees(drift_open)) / 5
    assert abs(closed.wz) < 0.05  # yaw rate settled near 0


def test_pid_turns_as_much_as_asked():
    # feed-forward even points the wrong way (left), the robot has inertia: the PID still gets -1.5 rad/s
    closed, plant, _, ft = make(calibrated(pid_enabled=True, kp=0.06, ki=0.8), gain=4.0, tau=0.1)
    mean_wz = run(closed, ft, 3.0, 0.1, 0.2, -1.5, tail=1.0)
    assert mean_wz == pytest.approx(-1.5, abs=0.15)
    open_loop, plant2, _, ft2 = make(calibrated(pid_enabled=False), gain=4.0, tau=0.1)
    assert run(open_loop, ft2, 3.0, 0.1, 0.2, -1.5, tail=1.0) > 0  # without PID it turns the wrong way


def test_pid_needs_calibrated_imu():
    ctl, _, _, ft = make(MotorTuning(pid_enabled=True, imu_calibrated=False))
    run(ctl, ft, 0.5, 0.3, 0.3, 0.0)
    assert ctl.u == 0.0


def test_start_power_and_kick():
    ctl, _, wheels, ft = make(MotorTuning(min_left=0.3, min_right=0.4, kick=0.7, kick_ms=100))
    ctl.command(0.05, 0.05, None)
    assert ctl.applied[0] == pytest.approx(0.7) and ctl.applied[1] == pytest.approx(0.7)  # kick
    for _ in range(15):  # 150 ms later: kick over, floors apply
        ft.sleep(0.01)
        with ctl._lock:
            ctl._apply_locked(ft.clock())
    assert ctl.applied[0] == pytest.approx(0.3 + 0.7 * 0.05)
    assert ctl.applied[1] == pytest.approx(0.4 + 0.6 * 0.05)
    ctl.command(0.0, 0.0)
    assert wheels.last == (0.0, 0.0) and ctl.applied == (0.0, 0.0)


def test_swap_sides():
    ctl, _, wheels, _ = make(MotorTuning(swap_sides=True))
    ctl.command(0.2, 0.6, None)
    l, r = wheels.last
    assert l > r > 0


def _wait_task(ctl, timeout=5.0):
    t0 = time.time()
    while ctl.task["state"] == "running" and time.time() - t0 < timeout:
        time.sleep(0.01)
    return ctl.task


def test_imu_calibrate_finds_bias_and_sign():
    ctl, plant, _, _ = make(MotorTuning(), bias=0.05, sign=-1.0)
    ctl.start_task("imu_calibrate")
    task = _wait_task(ctl)
    assert task["state"] == "done", task
    assert ctl.t.gyro_bias == pytest.approx(0.05, abs=1e-6)
    assert ctl.t.imu_sign == -1.0 and ctl.t.imu_calibrated
    assert not ctl._busy and ctl.applied == (0.0, 0.0)


def test_imu_calibrate_fails_when_gyro_sees_nothing():
    ctl, plant, _, _ = make(MotorTuning(), gain=0.0)
    ctl.start_task("imu_calibrate")
    assert _wait_task(ctl)["state"] == "failed"
    assert not ctl.t.imu_calibrated


def test_find_start_power():
    ctl, _, _, _ = make(calibrated(), stall=(0.45, 0.55))  # applied duty (incl. pwm_min) where wheels start
    ctl.start_task("find_start_power")
    task = _wait_task(ctl)
    assert task["state"] == "done", task
    # the commanded duty d maps to (60 + d*195)/255 applied, so the start points are found in command units
    for side, stall in (("left", 0.45), ("right", 0.55)):
        d_start = (stall * 255 - 60) / 195
        assert d_start <= task["result"][f"min_{side}"] <= d_start + 0.08


def test_commands_ignored_while_task_runs():
    ctl, _, _, _ = make(MotorTuning())
    ctl._busy = True
    ctl.command(0.5, 0.5, None)
    assert ctl.applied == (0.0, 0.0)
    ctl.stop()  # the watchdog's stop must not interrupt a self-test
    assert ctl._busy


@pytest.mark.parametrize("chip", ["MPU-6050", "MPU-9250", "ICM-20948"])
def test_gyro_driver(chip):
    bus = FakeIMURegisters(chip)
    g = GyroZ(bus, sleep=lambda s: None)
    assert g.chip == chip
    bus.dps = 90.0
    assert g.read_wz() == pytest.approx(math.pi / 2, abs=1e-3)
    bus.dps = -45.0
    assert g.read_wz() == pytest.approx(-math.pi / 4, abs=1e-3)


def test_gyro_driver_rejects_unknown_chip():
    class Empty:
        def read_byte_data(self, a, r):
            return 0x00

        def write_byte_data(self, a, r, v):
            pass

    with pytest.raises(RuntimeError):
        GyroZ(Empty(), sleep=lambda s: None)


def test_robot_io_configure_and_drive_status_over_tcp(monkeypatch):
    zmq = pytest.importorskip("zmq")  # noqa: F841
    from duckie_yolo.robot import robot_io
    from duckie_yolo.robot.hw.camera import MockCamera
    from duckie_yolo.robot.real_env import RealRobotEnv

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(robot_io.signal, "signal", lambda *a, **k: None)
    wheels, _, _ = fake_wheels("v3")
    drive = DriveController(wheels, Plant(wheels))
    cam = MockCamera(64, 48, 60).start()
    args = argparse.Namespace(bind="127.0.0.1", port=port, watchdog=0.3, frame_timeout=0.2, max_duty=0.8)
    threading.Thread(target=lambda: robot_io.serve(args, wheels, cam, drive), daemon=True).start()
    try:
        env = RealRobotEnv("127.0.0.1", port, timeout_ms=3000)
        env.reset()
        r = env.configure({"min_left": 0.25, "kick": 0.0, "bogus": 1})
        assert r["ok"] and r["tuning"]["min_left"] == 0.25 and "bogus" not in r["tuning"]
        env.step([0.2, 0.5])
        d = env.last_md["drive"]
        assert d["wref"] == pytest.approx(0.5) and d["imu"] == "FAKE"
        assert env.drive_task("nope")["ok"] is False
        r = env.drive_task("imu_calibrate")
        assert r["ok"] and r["task"]["name"] == "imu_calibrate"
        env.close()
    finally:
        drive.close()
        cam.stop()
