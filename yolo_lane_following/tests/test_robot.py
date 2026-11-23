import argparse
import socket
import threading
import time

import numpy as np
import pytest

from duckie_yolo.robot.hw.motors import BACKWARD, FORWARD, RELEASE, WheelPWMConfig, duty_to_command, fake_wheels
from duckie_yolo.robot.hw.pca9685 import FULL, MODE1, PRESCALE, PCA9685, FakeSMBus
from duckie_yolo.robot.protocol import decode_frame, encode_frame

ADDR = 0x60


def test_pca9685_init_and_prescale():
    bus = FakeSMBus()
    PCA9685(bus, ADDR, 1600.0, sleep=lambda s: None)
    assert PCA9685.prescale_for(1600) == 3  # 25 MHz / 4096 / 1600 - 1 = 2.81 -> 3
    assert bus.regs[(ADDR, PRESCALE)] == 3
    assert bus.regs[(ADDR, MODE1)] & 0x80  # restarted
    assert not bus.regs[(ADDR, MODE1)] & 0x10  # awake


def test_pca9685_channel_registers():
    bus = FakeSMBus()
    pca = PCA9685(bus, ADDR, 1600.0, sleep=lambda s: None)
    pca.set_pwm(8, 0, 2000)
    assert bus.channel(ADDR, 8) == (0, 2000)
    pca.set_pin(10, True)
    assert bus.channel(ADDR, 10) == (FULL, 0)
    pca.set_pin(10, False)
    assert bus.channel(ADDR, 10) == (0, FULL)


def test_duty_to_pwm_mapping():
    cfg = WheelPWMConfig(60, 255, 0.01)
    assert duty_to_command(0.0, cfg) == (RELEASE, 0)
    assert duty_to_command(0.005, cfg) == (RELEASE, 0)
    assert duty_to_command(1.0, cfg) == (FORWARD, 255)
    assert duty_to_command(-1.0, cfg) == (BACKWARD, 255)
    assert duty_to_command(0.5, cfg) == (FORWARD, int(0.5 * 195 + 60))
    assert duty_to_command(float("nan"), cfg) == (RELEASE, 0)


def test_hatv3_wheels_signals():
    wheels, bus, gpio = fake_wheels("v3")
    wheels.set(0.5, -0.5)
    # left motor: direction on PCA9685 channels 10 (in1) / 9 (in2), speed on 8
    assert bus.channel(ADDR, 10) == (FULL, 0) and bus.channel(ADDR, 9) == (0, FULL)
    assert bus.channel(ADDR, 8) == (0, int(0.5 * 195 + 60) * 16)
    # right motor: direction on Jetson GPIO BOARD 33 (in1) / 31 (in2), speed on channel 13
    assert gpio.pins[33] == 0 and gpio.pins[31] == 1  # backward
    assert bus.channel(ADDR, 13) == (0, int(0.5 * 195 + 60) * 16)
    wheels.stop()
    assert bus.channel(ADDR, 8) == (0, 0) and bus.channel(ADDR, 13) == (0, 0)
    assert gpio.pins[33] == 1 and gpio.pins[31] == 1  # released


def test_wheels_max_duty_and_invert():
    wheels, bus, gpio = fake_wheels("v3", max_duty=0.5, invert_left=True)
    l, r = wheels.set(1.0, 1.0)
    assert l < 0 and r > 0
    assert abs(r) <= (0.5 * 195 + 60) / 255 + 1e-9


@pytest.mark.parametrize("enc", ["raw", "jpeg"])
def test_frame_encoding_roundtrip(enc):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :32] = (0, 200, 255)
    md, payload = encode_frame(img, enc, 95)
    out = decode_frame(md, payload)
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).mean() < (0.5 if enc == "raw" else 4)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_robot_io_end_to_end_with_watchdog():
    """robot_io (mock camera + fake motors) <-> RealRobotEnv over real TCP."""
    pytest.importorskip("zmq")
    from duckie_yolo.robot import robot_io
    from duckie_yolo.robot.hw.camera import MockCamera
    from duckie_yolo.robot.real_env import RealRobotEnv

    port = _free_port()
    args = argparse.Namespace(
        bind="127.0.0.1", port=port, watchdog=0.3, frame_timeout=0.2, max_duty=0.8,
    )
    wheels, bus, gpio = fake_wheels("v3")
    cam = MockCamera(64, 48, 60).start()

    # run the server loop in a thread; signal handlers can only be installed in the main thread
    orig_signal = robot_io.signal.signal
    robot_io.signal.signal = lambda *a, **k: None
    t = threading.Thread(target=lambda: robot_io.serve(args, wheels, cam), daemon=True)
    t.start()
    try:
        env = RealRobotEnv("127.0.0.1", port, timeout_ms=3000)
        obs = env.reset()
        assert obs.shape == (48, 64, 3)
        obs, *_ = env.step([0.3, 0.0])
        assert wheels.last[0] > 0 and wheels.last[1] > 0
        seq1 = env.last_seq
        env.step([0.3, 0.0])
        assert env.last_seq > seq1  # always a fresh frame
        time.sleep(0.8)  # no commands -> watchdog releases the wheels
        assert wheels.last == (0.0, 0.0)
        env.close()
    finally:
        robot_io.signal.signal = orig_signal
        cam.stop()
