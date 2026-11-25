"""Yaw-rate gyro of the Duckiebot IMU (DB21M/DB21J/DB26: on the HAT's I2C bus at 0x68), no ROS.

Only the z gyro is needed: it measures how fast the robot turns, which closes the loop for
driving straight (robot/drive.py). The chip is identified by its WHO_AM_I register:
  * InvenSense MPU-6050 / 6500 / 9250 / 9255 / ICM-20602 / 20689 family (WHO_AM_I @ 0x75)
  * TDK ICM-20948 (WHO_AM_I @ 0x00 = 0xEA, banked registers)
Both are set to +-500 deg/s (65.5 LSB per deg/s) with a ~40 Hz low-pass filter.
"""
from __future__ import annotations

import math
import time
from typing import Optional

MPU_WHO_AM_I = 0x75
MPU_IDS = {0x68: "MPU-6050", 0x70: "MPU-6500", 0x71: "MPU-9250", 0x73: "MPU-9255", 0x12: "ICM-20602",
           0x98: "ICM-20689", 0x72: "MPU-6052C", 0x74: "MPU-6515"}
MPU_PWR_MGMT_1, MPU_CONFIG, MPU_GYRO_CONFIG, MPU_GYRO_ZOUT_H = 0x6B, 0x1A, 0x1B, 0x47

ICM_REG_BANK_SEL, ICM_WHO_AM_I, ICM_ID = 0x7F, 0x00, 0xEA
ICM_PWR_MGMT_1, ICM_PWR_MGMT_2, ICM_GYRO_ZOUT_H = 0x06, 0x07, 0x37
ICM_GYRO_CONFIG_1 = 0x01  # bank 2

LSB_PER_DPS = 65.5  # +-500 deg/s full scale


class GyroZ:
    """z-axis gyro in rad/s (counter-clockwise seen from above = positive, after ``sign``)."""

    def __init__(self, bus, address: int = 0x68, sleep=time.sleep):
        self.bus = bus
        self.address = address
        self._sleep = sleep
        self.chip = self._detect()
        if self.chip is None:
            raise RuntimeError(f"no supported IMU at 0x{address:02x}")
        if self.chip == "ICM-20948":
            self._init_icm()
            self._reg = ICM_GYRO_ZOUT_H
        else:
            self._init_mpu()
            self._reg = MPU_GYRO_ZOUT_H

    @classmethod
    def open(cls, busnum: int, address: int = 0x68) -> "GyroZ":
        from smbus2 import SMBus

        return cls(SMBus(busnum), address)

    def _detect(self) -> Optional[str]:
        try:
            who = self.bus.read_byte_data(self.address, MPU_WHO_AM_I)
        except OSError:
            who = None
        if who in MPU_IDS:
            return MPU_IDS[who]
        try:
            self.bus.write_byte_data(self.address, ICM_REG_BANK_SEL, 0x00)
            if self.bus.read_byte_data(self.address, ICM_WHO_AM_I) == ICM_ID:
                return "ICM-20948"
        except OSError:
            pass
        return None

    def _w(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.address, reg, val & 0xFF)

    def _init_mpu(self) -> None:
        self._w(MPU_PWR_MGMT_1, 0x01)  # wake up, PLL with the x gyro as clock
        self._sleep(0.05)
        self._w(MPU_CONFIG, 0x03)  # digital low-pass ~44 Hz
        self._w(MPU_GYRO_CONFIG, 0x08)  # +-500 deg/s

    def _init_icm(self) -> None:
        self._w(ICM_REG_BANK_SEL, 0x00)
        self._w(ICM_PWR_MGMT_1, 0x01)  # wake up, auto clock
        self._sleep(0.05)
        self._w(ICM_PWR_MGMT_2, 0x00)  # accel + gyro on
        self._w(ICM_REG_BANK_SEL, 0x20)  # bank 2
        self._w(ICM_GYRO_CONFIG_1, (3 << 3) | (1 << 1) | 1)  # DLPF cfg 3, +-500 deg/s, DLPF on
        self._w(ICM_REG_BANK_SEL, 0x00)

    def read_raw(self) -> int:
        hi, lo = self.bus.read_i2c_block_data(self.address, self._reg, 2)
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def read_wz(self) -> float:
        """Raw z rate in rad/s (no bias / sign correction: see drive.MotorTuning)."""
        return math.radians(self.read_raw() / LSB_PER_DPS)

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass


class FakeIMURegisters:
    """In-memory SMBus for an IMU (tests): register file + a settable z rate."""

    def __init__(self, chip: str = "MPU-6050"):
        self.regs = {}
        self.chip = chip
        self.dps = 0.0
        if chip == "ICM-20948":
            self.regs[ICM_WHO_AM_I] = ICM_ID
        else:
            who = {v: k for k, v in MPU_IDS.items()}[chip]
            self.regs[MPU_WHO_AM_I] = who

    def read_byte_data(self, addr, reg):
        if self.chip == "ICM-20948" and reg == MPU_WHO_AM_I:
            return 0x00
        return self.regs.get(reg, 0)

    def write_byte_data(self, addr, reg, val):
        self.regs[reg] = val

    def read_i2c_block_data(self, addr, reg, n):
        raw = int(round(self.dps * LSB_PER_DPS))
        raw = max(-32768, min(32767, raw)) & 0xFFFF
        return [raw >> 8, raw & 0xFF][:n]

    def close(self):
        pass
