"""Reusable control library for cable-driven robots using DYNAMIXEL XM430.

The public command convention is:

    positive displacement_mm -> shorten / wind the cable
    negative displacement_mm -> release / unwind the cable

Displacements are absolute values relative to the configured zero position.
By default, the motor positions measured during ``start()`` become 0 mm.

Typical use::

    from dynamixel_cable import CableController, CableControllerConfig

    config = CableControllerConfig(...)
    with CableController(config) as robot:
        robot.set_displacements_mm([1.0, 0, 0, -1.0, 0, 0], wait=True)
        robot.move_to_zero(wait=True)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)


COUNTS_PER_REVOLUTION = 4096
PROTOCOL_VERSION = 2.0

# XM430-W210 control-table addresses.
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4

MODE_EXTENDED_POSITION = 4
TORQUE_OFF = 0
TORQUE_ON = 1

EXTENDED_POSITION_MIN = -1_048_575
EXTENDED_POSITION_MAX = 1_048_575


class CableControllerError(RuntimeError):
    """Base exception raised by the cable controller."""


class CableControllerTimeout(CableControllerError):
    """Raised when the motors do not reach a requested position in time."""


@dataclass(frozen=True)
class CableControllerConfig:
    """Configuration for a group of cable-driving DYNAMIXEL motors.

    ``cable_mm_per_motor_rev`` should preferably be measured experimentally
    for each winch. For a direct-mounted constant-radius drum it is
    approximately ``2 * pi * drum_radius_mm``.

    ``wind_direction`` must contain +1 or -1 for each motor:
      +1 -> increasing encoder count shortens the cable
      -1 -> decreasing encoder count shortens the cable
    """

    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 57_600
    motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    cable_mm_per_motor_rev: tuple[float, ...] = (37.699,) * 6
    wind_direction: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    min_displacement_mm: tuple[float, ...] = (-2.0,) * 6
    max_displacement_mm: tuple[float, ...] = (2.0,) * 6
    profile_velocity: int = 20
    profile_acceleration: int = 10
    protocol_version: float = PROTOCOL_VERSION
    verbose: bool = False

    def __post_init__(self) -> None:
        count = len(self.motor_ids)
        if count == 0:
            raise ValueError("At least one motor ID is required")

        if len(set(self.motor_ids)) != count:
            raise ValueError("Motor IDs must be unique")

        parallel_fields = {
            "cable_mm_per_motor_rev": self.cable_mm_per_motor_rev,
            "wind_direction": self.wind_direction,
            "min_displacement_mm": self.min_displacement_mm,
            "max_displacement_mm": self.max_displacement_mm,
        }
        for name, values in parallel_fields.items():
            if len(values) != count:
                raise ValueError(
                    f"{name} must contain exactly {count} values"
                )

        for index, mm_per_rev in enumerate(self.cable_mm_per_motor_rev):
            if not math.isfinite(mm_per_rev) or mm_per_rev <= 0:
                raise ValueError(
                    f"cable_mm_per_motor_rev[{index}] must be positive"
                )

        for index, direction in enumerate(self.wind_direction):
            if direction not in (-1, 1):
                raise ValueError(
                    f"wind_direction[{index}] must be either +1 or -1"
                )

        for index, (minimum, maximum) in enumerate(
            zip(self.min_displacement_mm, self.max_displacement_mm)
        ):
            if minimum > maximum:
                raise ValueError(
                    f"Cable {index + 1}: minimum displacement exceeds maximum"
                )

        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if self.profile_velocity < 0:
            raise ValueError("profile_velocity cannot be negative")
        if self.profile_acceleration < 0:
            raise ValueError("profile_acceleration cannot be negative")

    @classmethod
    def from_drum_radii(
        cls,
        *,
        drum_radius_mm: Sequence[float],
        **kwargs: object,
    ) -> "CableControllerConfig":
        """Create a configuration from constant drum radii in millimetres."""
        travel = tuple(2.0 * math.pi * float(radius) for radius in drum_radius_mm)
        return cls(cable_mm_per_motor_rev=travel, **kwargs)


class CableController:
    """Synchronous position controller for cable-driving DYNAMIXEL motors.

    Use ``start()`` before commanding motors and ``close()`` afterwards, or
    use the class as a context manager. The context manager automatically
    calls ``start()`` and disables torque when leaving the block.
    """

    def __init__(self, config: CableControllerConfig) -> None:
        self.config = config
        self.port = PortHandler(config.device_name)
        self.packet = PacketHandler(config.protocol_version)
        self.sync_write = GroupSyncWrite(
            self.port,
            self.packet,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        self.zero_counts: list[int] | None = None
        self.commanded_displacements_mm: list[float] = [
            0.0 for _ in config.motor_ids
        ]
        self.connected = False
        self.initialised = False
        self._lock = threading.RLock()

    @property
    def motor_count(self) -> int:
        return len(self.config.motor_ids)

    def __enter__(self) -> "CableController":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(disable_torque=True)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)

    @staticmethod
    def _unsigned_to_signed32(value: int) -> int:
        return value - (1 << 32) if value & 0x80000000 else value

    @staticmethod
    def _signed32_to_bytes(value: int) -> list[int]:
        return list((value & 0xFFFFFFFF).to_bytes(4, "little", signed=False))

    def _check(
        self,
        motor_id: int,
        communication_result: int,
        motor_error: int,
    ) -> None:
        if communication_result != COMM_SUCCESS:
            detail = self.packet.getTxRxResult(communication_result)
            raise CableControllerError(f"Motor {motor_id}: {detail}")
        if motor_error != 0:
            detail = self.packet.getRxPacketError(motor_error)
            raise CableControllerError(f"Motor {motor_id}: {detail}")

    def _require_connected(self) -> None:
        if not self.connected:
            raise CableControllerError("Controller is not connected")

    def _require_initialised(self) -> None:
        if not self.initialised or self.zero_counts is None:
            raise CableControllerError(
                "Controller is not initialised; call start() or initialise()"
            )

    def _write1(self, motor_id: int, address: int, value: int) -> None:
        result, error = self.packet.write1ByteTxRx(
            self.port, motor_id, address, value
        )
        self._check(motor_id, result, error)

    def _write4(self, motor_id: int, address: int, value: int) -> None:
        result, error = self.packet.write4ByteTxRx(
            self.port, motor_id, address, value
        )
        self._check(motor_id, result, error)

    def _read1(self, motor_id: int, address: int) -> int:
        value, result, error = self.packet.read1ByteTxRx(
            self.port, motor_id, address
        )
        self._check(motor_id, result, error)
        return int(value)

    def _read_position_count(self, motor_id: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(
            self.port, motor_id, ADDR_PRESENT_POSITION
        )
        self._check(motor_id, result, error)
        return self._unsigned_to_signed32(int(value))

    def connect(self) -> None:
        """Open the serial port and confirm that all configured motors respond."""
        with self._lock:
            if self.connected:
                return

            if not self.port.openPort():
                raise CableControllerError(
                    f"Could not open serial port {self.config.device_name}"
                )

            try:
                if not self.port.setBaudRate(self.config.baudrate):
                    raise CableControllerError(
                        f"Could not set baud rate {self.config.baudrate}"
                    )

                for motor_id in self.config.motor_ids:
                    model, result, error = self.packet.ping(self.port, motor_id)
                    self._check(motor_id, result, error)
                    self._log(
                        f"Found motor ID {motor_id}, model number {model}"
                    )

                self.connected = True
            except Exception:
                self.port.closePort()
                raise

    def initialise(self, *, use_present_position_as_zero: bool = True) -> None:
        """Configure Extended Position Mode and enable torque safely.

        When ``use_present_position_as_zero`` is True, the present physical
        motor positions become the 0 mm cable reference.
        """
        with self._lock:
            self._require_connected()

            for motor_id in self.config.motor_ids:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

            for motor_id in self.config.motor_ids:
                mode = self._read1(motor_id, ADDR_OPERATING_MODE)
                if mode != MODE_EXTENDED_POSITION:
                    self._write1(
                        motor_id,
                        ADDR_OPERATING_MODE,
                        MODE_EXTENDED_POSITION,
                    )

                self._write4(
                    motor_id,
                    ADDR_PROFILE_ACCELERATION,
                    self.config.profile_acceleration,
                )
                self._write4(
                    motor_id,
                    ADDR_PROFILE_VELOCITY,
                    self.config.profile_velocity,
                )

            present_counts = self.get_present_counts()

            if use_present_position_as_zero or self.zero_counts is None:
                self.zero_counts = present_counts.copy()
                self.commanded_displacements_mm = [0.0] * self.motor_count

            # Set current position as goal before torque is enabled. This avoids
            # an unexpected movement caused by a stale Goal Position register.
            self._sync_goal_positions(present_counts)

            for motor_id in self.config.motor_ids:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

            self.initialised = True
            self._log(f"Zero counts: {self.zero_counts}")

    def start(self, *, use_present_position_as_zero: bool = True) -> None:
        """Connect, configure the motors and enable torque."""
        self.connect()
        self.initialise(
            use_present_position_as_zero=use_present_position_as_zero
        )

    def enable_torque(self) -> None:
        with self._lock:
            self._require_connected()
            for motor_id in self.config.motor_ids:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

    def disable_torque(self) -> None:
        with self._lock:
            if not self.connected:
                return
            errors: list[str] = []
            for motor_id in self.config.motor_ids:
                try:
                    self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
                except Exception as exc:  # Attempt every motor before failing.
                    errors.append(str(exc))
            self.initialised = False
            if errors:
                raise CableControllerError("; ".join(errors))

    def close(self, *, disable_torque: bool = True) -> None:
        """Optionally disable torque and close the serial port."""
        with self._lock:
            if not self.connected:
                return

            torque_error: Exception | None = None
            if disable_torque:
                try:
                    self.disable_torque()
                except Exception as exc:
                    torque_error = exc

            self.port.closePort()
            self.connected = False
            self.initialised = False

            if torque_error is not None:
                raise torque_error

    def set_zero_from_present_position(self) -> list[int]:
        """Redefine the present physical state as 0 mm without moving."""
        with self._lock:
            self._require_connected()
            self.zero_counts = self.get_present_counts()
            self.commanded_displacements_mm = [0.0] * self.motor_count
            return self.zero_counts.copy()

    def get_present_counts(self) -> list[int]:
        """Read all present motor positions in encoder counts."""
        with self._lock:
            self._require_connected()
            return [
                self._read_position_count(motor_id)
                for motor_id in self.config.motor_ids
            ]

    def counts_to_displacements_mm(
        self,
        counts: Sequence[int],
    ) -> list[float]:
        """Convert absolute encoder counts to cable displacement in mm."""
        self._require_initialised()
        if len(counts) != self.motor_count:
            raise ValueError(
                f"Expected {self.motor_count} count values, got {len(counts)}"
            )

        assert self.zero_counts is not None
        displacements: list[float] = []
        for index, count in enumerate(counts):
            delta_counts = int(count) - self.zero_counts[index]
            displacement = (
                delta_counts
                * self.config.wind_direction[index]
                * self.config.cable_mm_per_motor_rev[index]
                / COUNTS_PER_REVOLUTION
            )
            displacements.append(displacement)
        return displacements

    def displacements_mm_to_counts(
        self,
        displacements_mm: Sequence[float],
    ) -> list[int]:
        """Convert cable displacement targets in mm to absolute counts."""
        self._require_initialised()
        values = self._validate_displacements(displacements_mm)
        assert self.zero_counts is not None

        goals: list[int] = []
        for index, displacement in enumerate(values):
            delta_counts = round(
                displacement
                * COUNTS_PER_REVOLUTION
                / self.config.cable_mm_per_motor_rev[index]
            )
            goal = (
                self.zero_counts[index]
                + self.config.wind_direction[index] * delta_counts
            )
            if not EXTENDED_POSITION_MIN <= goal <= EXTENDED_POSITION_MAX:
                raise ValueError(
                    f"Motor {self.config.motor_ids[index]}: goal {goal} counts "
                    "is outside the Extended Position Mode range"
                )
            goals.append(goal)
        return goals

    def get_present_displacements_mm(self) -> list[float]:
        """Read the current cable displacement of every motor in millimetres."""
        with self._lock:
            counts = self.get_present_counts()
            return self.counts_to_displacements_mm(counts)

    def _validate_displacements(
        self,
        displacements_mm: Sequence[float],
    ) -> list[float]:
        if len(displacements_mm) != self.motor_count:
            raise ValueError(
                f"Expected {self.motor_count} cable displacements, "
                f"got {len(displacements_mm)}"
            )

        values: list[float] = []
        for index, raw_value in enumerate(displacements_mm):
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Cable {index + 1}: displacement must be finite"
                )

            minimum = self.config.min_displacement_mm[index]
            maximum = self.config.max_displacement_mm[index]
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Cable {index + 1}: {value:.3f} mm is outside "
                    f"[{minimum:.3f}, {maximum:.3f}] mm"
                )
            values.append(value)
        return values

    def _sync_goal_positions(self, goals: Sequence[int]) -> None:
        if len(goals) != self.motor_count:
            raise ValueError(
                f"Expected {self.motor_count} goal positions, got {len(goals)}"
            )

        self.sync_write.clearParam()
        try:
            for motor_id, goal in zip(self.config.motor_ids, goals):
                if not EXTENDED_POSITION_MIN <= int(goal) <= EXTENDED_POSITION_MAX:
                    raise ValueError(
                        f"Motor {motor_id}: goal {goal} is outside the "
                        "Extended Position Mode range"
                    )

                added = self.sync_write.addParam(
                    motor_id,
                    self._signed32_to_bytes(int(goal)),
                )
                if not added:
                    raise CableControllerError(
                        f"Could not add motor {motor_id} to GroupSyncWrite"
                    )

            result = self.sync_write.txPacket()
            if result != COMM_SUCCESS:
                raise CableControllerError(self.packet.getTxRxResult(result))
        finally:
            self.sync_write.clearParam()

    def set_displacements_mm(
        self,
        displacements_mm: Sequence[float],
        *,
        wait: bool = False,
        tolerance_mm: float = 0.2,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> None:
        """Command all cable displacements relative to the startup zero.

        The command is sent to all motors in one GroupSyncWrite packet.
        Set ``wait=True`` to block until every cable is within ``tolerance_mm``
        of its target or until ``timeout_s`` is exceeded.
        """
        with self._lock:
            self._require_initialised()
            values = self._validate_displacements(displacements_mm)
            goals = self.displacements_mm_to_counts(values)
            self._sync_goal_positions(goals)
            self.commanded_displacements_mm = values.copy()

        if wait:
            self.wait_until_reached(
                target_displacements_mm=values,
                tolerance_mm=tolerance_mm,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )

    def set_cable_mm(
        self,
        cable_number: int,
        displacement_mm: float,
        *,
        wait: bool = False,
        tolerance_mm: float = 0.2,
        timeout_s: float = 5.0,
    ) -> None:
        """Update one cable target while preserving the other targets.

        ``cable_number`` is one-based: cable 1 corresponds to the first motor
        ID in the configuration.
        """
        if not 1 <= cable_number <= self.motor_count:
            raise ValueError(
                f"cable_number must be between 1 and {self.motor_count}"
            )

        targets = self.commanded_displacements_mm.copy()
        targets[cable_number - 1] = float(displacement_mm)
        self.set_displacements_mm(
            targets,
            wait=wait,
            tolerance_mm=tolerance_mm,
            timeout_s=timeout_s,
        )

    def move_to_zero(
        self,
        *,
        wait: bool = False,
        tolerance_mm: float = 0.2,
        timeout_s: float = 5.0,
    ) -> None:
        """Move every cable to the 0 mm reference position."""
        self.set_displacements_mm(
            [0.0] * self.motor_count,
            wait=wait,
            tolerance_mm=tolerance_mm,
            timeout_s=timeout_s,
        )

    def wait_until_reached(
        self,
        *,
        target_displacements_mm: Sequence[float] | None = None,
        tolerance_mm: float = 0.2,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> list[float]:
        """Wait until every motor reaches its cable target within tolerance."""
        if tolerance_mm < 0:
            raise ValueError("tolerance_mm cannot be negative")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

        if target_displacements_mm is None:
            targets = self.commanded_displacements_mm.copy()
        else:
            targets = self._validate_displacements(target_displacements_mm)

        deadline = time.monotonic() + timeout_s
        while True:
            present = self.get_present_displacements_mm()
            errors = [
                abs(measured - target)
                for measured, target in zip(present, targets)
            ]
            if all(error <= tolerance_mm for error in errors):
                return present

            if time.monotonic() >= deadline:
                formatted = ", ".join(f"{error:.3f}" for error in errors)
                raise CableControllerTimeout(
                    "Target was not reached within "
                    f"{timeout_s:.2f} s; cable errors [mm]: [{formatted}]"
                )

            time.sleep(poll_interval_s)


__all__ = [
    "CableController",
    "CableControllerConfig",
    "CableControllerError",
    "CableControllerTimeout",
    "COUNTS_PER_REVOLUTION",
]