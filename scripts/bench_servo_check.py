"""Bench check: can one servo move 200 units from wherever it is parked.

Edit PORT and SERVO_ID below, then:

    python bench_servo_check.py
"""

import time

from cartesian_hand.servo import open_driver as FtServo

PORT = "/dev/ttyACM0"
SERVO_ID = 20
MOVE_COUNTS = 200
SPEED = 300
ACC = 50
TORQUE = 400  # force cap, 0-1000; raise if the DOF is loaded (e.g. z against gravity)

drv = FtServo(PORT)

start = drv.read_position(SERVO_ID)
if start is None:
    raise SystemExit(f"no response from ID {SERVO_ID} on {PORT}")
print(f"ID {SERVO_ID} at {start}")

drv.enable_torque(SERVO_ID, True)
try:
    goal = start + MOVE_COUNTS
    print(f"moving to {goal}")
    drv.set_position(SERVO_ID, goal, SPEED, ACC, TORQUE)
    time.sleep(1.0)

    moved = drv.read_position(SERVO_ID)
    print(f"now at {moved}")

    print(f"returning to {start}")
    drv.set_position(SERVO_ID, start, SPEED, ACC, TORQUE)
    time.sleep(1.0)
    print(f"back at {drv.read_position(SERVO_ID)}")
finally:
    drv.enable_torque(SERVO_ID, False)
    drv.close()

if moved is None:
    print("FAIL: lost contact with the servo mid-move")
elif abs(moved - start) > MOVE_COUNTS / 2:
    print(f"PASS: moved {moved - start} counts")
else:
    print(f"FAIL: only moved {moved - start} counts, expected ~{MOVE_COUNTS}")
