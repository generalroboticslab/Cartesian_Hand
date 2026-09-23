"""Move every servo on the bus 5mm out and back.

    python scripts/ft_servo_tools/example.py /dev/ttyACM0

Every call here is a batch call. One servo is the same call with a length-1
list. Timings and the rest of the API are in README.md.
"""

import sys
import time

from cartesian_hand.servo import open_driver

NUDGE = 600  # counts, ~5mm on a 16mm rack

if len(sys.argv) < 2:
    sys.exit(__doc__)

drv = open_driver(sys.argv[1])
ids = drv.scan(0, 20)
start = drv.read_positions(ids)  # None per servo that did not answer
print(f"{ids} at {start}")

# Command where they already are before energizing, or each servo snaps to
# whatever goal is left in its register from last time.
drv.set_positions(ids, start, speed=0, acc=50, torque=50)
drv.enable_torques(ids, True)

try:
    drv.set_positions(ids, [c + NUDGE for c in start], speed=300, acc=50, torque=50)
    time.sleep(1.0)
    print(f"moved to {drv.read_positions(ids)}")

    drv.set_positions(ids, start, speed=300, acc=50, torque=50)
    time.sleep(1.0)
    print(f"back at {drv.read_positions(ids)}")
finally:
    drv.enable_torques(ids, False)  # close() does not do this
    drv.close()
