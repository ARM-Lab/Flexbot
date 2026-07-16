"""Minimal example of using dynamixel_cable.py from another Python file."""

from dynamixel_cable import CableController, CableControllerConfig


# Replace 37.699 with the measured cable travel for one complete output-shaft
# revolution of each motor. Also calibrate every winding direction.
CONFIG = CableControllerConfig(
    device_name="/dev/ttyUSB0",
    baudrate=57600,
    motor_ids=(1, 2, 3, 4, 5, 6),
    cable_mm_per_motor_rev=(37.699, 37.699, 37.699, 37.699, 37.699, 37.699),
    wind_direction=(1, 1, 1, 1, 1, 1),
    min_displacement_mm=(-2.0, -2.0, -2.0, -2.0, -2.0, -2.0),
    max_displacement_mm=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
    profile_velocity=20,
    profile_acceleration=10,
    verbose=True,
)


def main() -> None:
    # Place the robot in its known straight, pretensioned state before this
    # block begins. start() uses the present motor positions as 0 mm.
    with CableController(CONFIG) as robot:
        robot.set_displacements_mm(
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            wait=True,
            tolerance_mm=0.2,
            timeout_s=5.0,
        )

        measured = robot.get_present_displacements_mm()
        print("Measured cable displacements [mm]:", measured)

        # Change only cable 3; the targets of the other cables are preserved.
        robot.set_cable_mm(3, 0.5, wait=True)

        # Return all six motors to the startup-zero positions.
        robot.move_to_zero(wait=True)


if __name__ == "__main__":
    main()