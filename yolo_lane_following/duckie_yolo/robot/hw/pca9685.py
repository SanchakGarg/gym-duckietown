"""Minimal register-level driver for the NXP PCA9685 16-channel PWM controller.

The Duckietown HAT (all versions) drives the motor H-bridge through a PCA9685. This driver
talks to it directly over I2C with ``smbus2`` - no ROS, no Duckietown containers.
Register sequence follows the PCA9685 datasheet (and the classic Adafruit driver).
"""
from __future__ import annotations

import math
import time
from typing import Optional

MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06
ALL_LED_ON_L = 0xFA

RESTART = 0x80
SLEEP = 0x10
ALLCALL = 0x01
OUTDRV = 0x04

OSC_HZ = 25_000_000
FULL = 4096  # value with the "full on / full off" bit set


class PCA9685:
    def __init__(self, bus, address: int = 0x60, frequency: float = 1600.0, sleep=time.sleep):
        """``bus`` is an smbus2.SMBus-compatible object (write_byte_data / read_byte_data)."""
        self.bus = bus
        self.address = address
        self._sleep = sleep
        self.set_all_pwm(0, 0)
        self._write(MODE2, OUTDRV)
        self._write(MODE1, ALLCALL)
        self._sleep(0.005)
        mode1 = self._read(MODE1) & ~SLEEP  # wake up
        self._write(MODE1, mode1)
        self._sleep(0.005)
        self.set_frequency(frequency)

    @classmethod
    def open(cls, busnum: int = 7, address: int = 0x60, frequency: float = 1600.0) -> "PCA9685":
        from smbus2 import SMBus

        return cls(SMBus(busnum), address, frequency)

    # ---------------------------------------------------------------- registers
    def _write(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _read(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    @staticmethod
    def prescale_for(frequency: float) -> int:
        return int(math.floor(OSC_HZ / 4096.0 / float(frequency) - 1.0 + 0.5))

    def set_frequency(self, frequency: float) -> None:
        prescale = self.prescale_for(frequency)
        old = self._read(MODE1)
        self._write(MODE1, (old & 0x7F) | SLEEP)  # prescale can only be set while sleeping
        self._write(PRESCALE, prescale)
        self._write(MODE1, old)
        self._sleep(0.005)
        self._write(MODE1, old | RESTART)

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        """on/off are 12-bit tick counts (0..4095); 4096 sets the full-on / full-off bit."""
        base = LED0_ON_L + 4 * int(channel)
        self._write(base, on & 0xFF)
        self._write(base + 1, on >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    def set_all_pwm(self, on: int, off: int) -> None:
        self._write(ALL_LED_ON_L, on & 0xFF)
        self._write(ALL_LED_ON_L + 1, on >> 8)
        self._write(ALL_LED_ON_L + 2, off & 0xFF)
        self._write(ALL_LED_ON_L + 3, off >> 8)

    def set_pin(self, channel: int, high: bool) -> None:
        """Use a PWM channel as a digital output."""
        if high:
            self.set_pwm(channel, FULL, 0)
        else:
            self.set_pwm(channel, 0, FULL)

    def close(self) -> None:
        try:
            self.set_all_pwm(0, FULL)
        finally:
            close = getattr(self.bus, "close", None)
            if close:
                close()


class FakeSMBus:
    """In-memory stand-in for smbus2.SMBus (tests / --mock runs)."""

    def __init__(self):
        self.regs = {}
        self.log = []

    def write_byte_data(self, addr: int, reg: int, value: int) -> None:
        self.regs[(addr, reg)] = value
        self.log.append((addr, reg, value))

    def read_byte_data(self, addr: int, reg: int) -> int:
        return self.regs.get((addr, reg), 0)

    def close(self) -> None:
        pass

    def channel(self, addr: int, ch: int) -> Optional[tuple]:
        base = LED0_ON_L + 4 * ch
        try:
            on = self.regs[(addr, base)] | (self.regs[(addr, base + 1)] << 8)
            off = self.regs[(addr, base + 2)] | (self.regs[(addr, base + 3)] << 8)
        except KeyError:
            return None
        return on, off
