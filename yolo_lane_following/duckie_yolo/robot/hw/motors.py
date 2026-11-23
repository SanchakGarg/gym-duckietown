"""Duckiebot wheel motors driven directly through the Duckietown HAT (no ROS).

Pin mapping and PWM conventions replicate Duckietown's ``hat_driver`` / ``wheels_driver``
(dt-duckiebot-interface, reference only - nothing is imported from it):

* HATv3 (DB21M / DB21J / DB26J, Jetson Nano / Orin Nano): PCA9685 @ 0x60, 1600 Hz
    left  motor: direction on PCA9685 channels in1=10, in2=9, speed on channel 8
    right motor: direction on Jetson GPIO BOARD pins in1=33, in2=31, speed on channel 13
  On the Orin Nano (DB26J) the PCA9685 sits on I2C bus 7; on the Jetson Nano on bus 1.
* HATv1/v2 (DB18/DB19/DB20, Raspberry Pi): both motors on the PCA9685
    left (10, 9, 8), right (11, 12, 13)

Speed: |duty| in [0, 1] -> 8-bit pwm = floor(|duty| * (pwm_max - pwm_min) + pwm_min) -> *16
(12-bit), with a dead zone around 0 where the motor is released.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .pca9685 import FULL, PCA9685

LOW, HIGH = 0, 1
RELEASE, FORWARD, BACKWARD = 0, 1, -1


@dataclass
class MotorPins:
    in1: int
    in2: int
    pwm: int
    gpio_direction: bool  # True: in1/in2 are Jetson/RPi GPIO pins (BOARD numbering)


HAT_PINS = {
    "v3": (MotorPins(10, 9, 8, False), MotorPins(33, 31, 13, True)),
    "v2": (MotorPins(10, 9, 8, False), MotorPins(11, 12, 13, False)),
    "v1": (MotorPins(10, 9, 8, False), MotorPins(11, 12, 13, False)),
}

# direction -> (in1, in2), as in dt-duckiebot-interface hat_driver/motor.py
PWM_DIR_SIGNALS = {RELEASE: (LOW, HIGH), FORWARD: (HIGH, LOW), BACKWARD: (LOW, HIGH)}
GPIO_DIR_SIGNALS = {RELEASE: (HIGH, HIGH), FORWARD: (HIGH, LOW), BACKWARD: (LOW, HIGH)}


@dataclass
class WheelPWMConfig:
    pwm_min: int = 60
    pwm_max: int = 255
    deadzone: float = 0.01


def duty_to_command(duty: float, cfg: WheelPWMConfig) -> Tuple[int, int]:
    """duty in [-1, 1] -> (direction, 8-bit speed)."""
    if not math.isfinite(duty) or abs(duty) < cfg.deadzone:
        return RELEASE, 0
    pwm = int(math.floor(min(abs(duty), 1.0) * (cfg.pwm_max - cfg.pwm_min) + cfg.pwm_min))
    return (FORWARD if duty > 0 else BACKWARD), min(pwm, cfg.pwm_max)


class GPIOBackend:
    """Jetson.GPIO (or RPi.GPIO) in BOARD mode."""

    def __init__(self):
        try:
            import Jetson.GPIO as GPIO  # type: ignore
        except ImportError:  # Raspberry Pi based Duckiebots
            import RPi.GPIO as GPIO  # type: ignore
        self.GPIO = GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

    def setup_output(self, pin: int) -> None:
        self.GPIO.setup(pin, self.GPIO.OUT, initial=self.GPIO.HIGH)

    def output(self, pin: int, value: int) -> None:
        self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)

    def cleanup(self, pins) -> None:
        try:
            self.GPIO.cleanup(list(pins))
        except Exception:
            pass


class FakeGPIO:
    def __init__(self):
        self.pins = {}

    def setup_output(self, pin: int) -> None:
        self.pins[pin] = HIGH

    def output(self, pin: int, value: int) -> None:
        self.pins[pin] = HIGH if value else LOW

    def cleanup(self, pins) -> None:
        pass


class HATMotor:
    def __init__(self, name: str, pca: PCA9685, pins: MotorPins, gpio=None):
        self.name = name
        self.pca = pca
        self.pins = pins
        self.gpio = gpio
        if pins.gpio_direction:
            if gpio is None:
                raise ValueError(f"motor {name}: GPIO direction pins need a GPIO backend")
            gpio.setup_output(pins.in1)
            gpio.setup_output(pins.in2)
        self.set(RELEASE, 0)

    def _direction(self, direction: int) -> None:
        if self.pins.gpio_direction:
            a, b = GPIO_DIR_SIGNALS[direction]
            self.gpio.output(self.pins.in1, a)
            self.gpio.output(self.pins.in2, b)
        else:
            a, b = PWM_DIR_SIGNALS[direction]
            self.pca.set_pin(self.pins.in1, bool(a))
            self.pca.set_pin(self.pins.in2, bool(b))

    def set(self, direction: int, speed: int) -> None:
        speed = max(0, min(int(speed), 255))
        self._direction(direction)
        self.pca.set_pwm(self.pins.pwm, 0, speed * 16)


class Wheels:
    """Left/right wheel duty cycles in [-1, 1]."""

    def __init__(
        self,
        pca: PCA9685,
        hat_version: str = "v3",
        gpio=None,
        pwm_cfg: Optional[WheelPWMConfig] = None,
        invert_left: bool = False,
        invert_right: bool = False,
        max_duty: float = 1.0,
    ):
        left_pins, right_pins = HAT_PINS[hat_version]
        self.pca = pca
        self.gpio = gpio
        self.cfg = pwm_cfg or WheelPWMConfig()
        self.invert = (-1.0 if invert_left else 1.0, -1.0 if invert_right else 1.0)
        self.max_duty = max_duty
        self.left = HATMotor("left", pca, left_pins, gpio)
        self.right = HATMotor("right", pca, right_pins, gpio)
        self._gpio_pins = [p for mp in (left_pins, right_pins) if mp.gpio_direction for p in (mp.in1, mp.in2)]
        self.last = (0.0, 0.0)

    @classmethod
    def open(cls, i2c_bus: int = 7, address: int = 0x60, hat_version: str = "v3", **kw) -> "Wheels":
        pca = PCA9685.open(i2c_bus, address, 1600.0)
        gpio = GPIOBackend() if hat_version == "v3" else None
        return cls(pca, hat_version, gpio, **kw)

    def set(self, left: float, right: float) -> Tuple[float, float]:
        lim = self.max_duty
        left = max(-lim, min(lim, float(left) * self.invert[0]))
        right = max(-lim, min(lim, float(right) * self.invert[1]))
        dl, sl = duty_to_command(left, self.cfg)
        dr, sr = duty_to_command(right, self.cfg)
        self.left.set(dl, sl)
        self.right.set(dr, sr)
        self.last = (dl * sl / 255.0, dr * sr / 255.0)
        return self.last

    def stop(self) -> None:
        self.left.set(RELEASE, 0)
        self.right.set(RELEASE, 0)
        self.last = (0.0, 0.0)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if self.gpio is not None:
                self.gpio.cleanup(self._gpio_pins)
            self.pca.close()


def fake_wheels(hat_version: str = "v3", **kw):
    """Wheels backed by in-memory I2C/GPIO (for tests and --mock runs)."""
    from .pca9685 import FakeSMBus

    bus = FakeSMBus()
    pca = PCA9685(bus, 0x60, 1600.0, sleep=lambda s: None)
    gpio = FakeGPIO()
    return Wheels(pca, hat_version, gpio, **kw), bus, gpio


__all__ = ["Wheels", "WheelPWMConfig", "duty_to_command", "fake_wheels", "HAT_PINS", "FULL"]
