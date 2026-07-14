"""
Control six DYNAMIXEL XM430-W210-R actuators as cable winches.
requirements: dynamixel sdk -python3 -m pip install dynamixel-sdk 

Command convention:
    positive displacement_mm = shorten / wind cable
    negative displacement_mm = release / unwind cable

The physical cable state at program startup is used as 0 mm.
Before running, set unique motor IDs and edit the configuration section.
"""

from typing import Sequence

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# ==================== USER CONFIGURATION ====================

DEVICE_NAME = "/dev/ttyUSB0"   # Windows example: "COM4"
BAUDRATE = 57600               # Factory default unless changed
PROTOCOL_VERSION = 2.0

DXL_IDS = [1, 2, 3, 4, 5, 6]

# Measured cable travel caused by ONE complete motor output-shaft revolution.
# Example: direct-mounted drum, effective radius 6 mm:
# 2*pi*6 mm = 37.699 mm/revolution.
#
# For best accuracy, measure this value experimentally for every winch.
CABLE_MM_PER_MOTOR_REV = [37.699] * 6

# Direction calibration:
# +1: increasing DYNAMIXEL position winds/shortens the cable
# -1: decreasing DYNAMIXEL position winds/shortens the cable
WIND_DIRECTION = [1, 1, 1, 1, 1, 1]

# Conservative commissioning limits relative to startup zero.
# Increase only after checking the mechanism and cable tension.
# Current value set to +/- 2 mm
MIN_DISPLACEMENT_MM = [-2.0] * 6
MAX_DISPLACEMENT_MM = [2.0] * 6

# Deliberately slow initial profile.
PROFILE_VELOCITY = 20
PROFILE_ACCELERATION = 10

# ============================================================

COUNTS_PER_REV = 4096

EXTENDED_POSITION_MIN = -1_048_575
EXTENDED_POSITION_MAX = 1_048_575

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


def unsigned_to_signed32(value: int) -> int:
    """Convert an unsigned 32-bit DYNAMIXEL register value to signed."""
    return value - (1 << 32) if value & 0x80000000 else value


def signed32_to_bytes(value: int) -> list[int]:
    """Encode a signed 32-bit position as little-endian bytes."""
    return list((value & 0xFFFFFFFF).to_bytes(4, "little", signed=False))


class CableController:
    def __init__(self) -> None:
        self.port = PortHandler(DEVICE_NAME)
        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.sync_write = GroupSyncWrite(
            self.port,
            self.packet,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self.zero_counts: list[int] = []

    def _check(self, dxl_id: int, comm_result: int, dxl_error: int) -> None:
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(
                f"Motor {dxl_id}: {self.packet.getTxRxResult(comm_result)}"
            )
        if dxl_error != 0:
            raise RuntimeError(
                f"Motor {dxl_id}: {self.packet.getRxPacketError(dxl_error)}"
            )

    def _write1(self, dxl_id: int, address: int, value: int) -> None:
        comm_result, dxl_error = self.packet.write1ByteTxRx(
            self.port, dxl_id, address, value
        )
        self._check(dxl_id, comm_result, dxl_error)

    def _write4(self, dxl_id: int, address: int, value: int) -> None:
        comm_result, dxl_error = self.packet.write4ByteTxRx(
            self.port, dxl_id, address, value
        )
        self._check(dxl_id, comm_result, dxl_error)

    def _read1(self, dxl_id: int, address: int) -> int:
        value, comm_result, dxl_error = self.packet.read1ByteTxRx(
            self.port, dxl_id, address
        )
        self._check(dxl_id, comm_result, dxl_error)
        return value

    def _read_position(self, dxl_id: int) -> int:
        value, comm_result, dxl_error = self.packet.read4ByteTxRx(
            self.port, dxl_id, ADDR_PRESENT_POSITION
        )
        self._check(dxl_id, comm_result, dxl_error)
        return unsigned_to_signed32(value)

    def connect(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Could not open {DEVICE_NAME}")

        if not self.port.setBaudRate(BAUDRATE):
            raise RuntimeError(f"Could not set baud rate {BAUDRATE}")

        for dxl_id in DXL_IDS:
            model, comm_result, dxl_error = self.packet.ping(
                self.port, dxl_id
            )
            self._check(dxl_id, comm_result, dxl_error)
            print(f"Found motor ID {dxl_id}, model number {model}")

    def _sync_goal_positions(self, goals: Sequence[int]) -> None:
        if len(goals) != len(DXL_IDS):
            raise ValueError("Exactly six goal positions are required")

        self.sync_write.clearParam()

        for dxl_id, goal in zip(DXL_IDS, goals):
            if not EXTENDED_POSITION_MIN <= goal <= EXTENDED_POSITION_MAX:
                raise ValueError(
                    f"Motor {dxl_id}: goal {goal} is outside "
                    "the extended-position range"
                )

            added = self.sync_write.addParam(
                dxl_id,
                signed32_to_bytes(int(goal)),
            )
            if not added:
                raise RuntimeError(
                    f"Could not add motor {dxl_id} to GroupSyncWrite"
                )

        comm_result = self.sync_write.txPacket()
        self.sync_write.clearParam()

        if comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(comm_result))

    def initialise(self) -> None:
        # Operating Mode is in EEPROM and can only be changed with torque off.
        for dxl_id in DXL_IDS:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

        for dxl_id in DXL_IDS:
            current_mode = self._read1(dxl_id, ADDR_OPERATING_MODE)

            if current_mode != MODE_EXTENDED_POSITION:
                self._write1(
                    dxl_id,
                    ADDR_OPERATING_MODE,
                    MODE_EXTENDED_POSITION,
                )

            self._write4(
                dxl_id,
                ADDR_PROFILE_ACCELERATION,
                PROFILE_ACCELERATION,
            )
            self._write4(
                dxl_id,
                ADDR_PROFILE_VELOCITY,
                PROFILE_VELOCITY,
            )

        # Use the present physical state as the 0 mm reference.
        self.zero_counts = [
            self._read_position(dxl_id) for dxl_id in DXL_IDS
        ]

        # Set the current positions as goals before enabling torque,
        # reducing the chance of an unexpected jump.
        self._sync_goal_positions(self.zero_counts)

        for dxl_id in DXL_IDS:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

        print("Startup zero counts:", self.zero_counts)

    def set_cable_displacements_mm(
        self,
        displacements_mm: Sequence[float],
    ) -> None:
        if len(displacements_mm) != len(DXL_IDS):
            raise ValueError("Enter exactly six cable displacements")

        goals: list[int] = []

        for index, displacement_mm in enumerate(displacements_mm):
            minimum = MIN_DISPLACEMENT_MM[index]
            maximum = MAX_DISPLACEMENT_MM[index]

            if not minimum <= displacement_mm <= maximum:
                raise ValueError(
                    f"Cable {index + 1}: {displacement_mm:.3f} mm "
                    f"is outside [{minimum}, {maximum}] mm"
                )

            delta_counts = round(
                displacement_mm
                * COUNTS_PER_REV
                / CABLE_MM_PER_MOTOR_REV[index]
            )

            goal = (
                self.zero_counts[index]
                + WIND_DIRECTION[index] * delta_counts
            )
            goals.append(goal)

        self._sync_goal_positions(goals)

        print(
            "Commanded displacement [mm]:",
            [round(value, 3) for value in displacements_mm],
        )
        print("Goal positions [counts]:   ", goals)

    def shutdown(self) -> None:
        # Support the robot mechanically before torque is removed.
        for dxl_id in DXL_IDS:
            try:
                self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except Exception as exc:
                print(
                    f"Could not disable torque on motor {dxl_id}: {exc}"
                )

        self.port.closePort()


def main() -> None:
    controller = CableController()

    try:
        controller.connect()
        controller.initialise()

        print(
            "\nEnter six cable displacements in millimetres.\n"
            "Positive = shorten/wind; negative = release/unwind.\n"
            "Example: 1.0 0 0 -1.0 0 0\n"
            "Enter q to stop.\n"
        )

        while True:
            command = input("cable_mm> ").strip()

            if command.lower() in {"q", "quit", "exit"}:
                break

            values = [
                float(value)
                for value in command.replace(",", " ").split()
            ]
            controller.set_cable_displacements_mm(values)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()