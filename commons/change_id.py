"""
change_id.py — scan bus, expect exactly 1 servo, rename it to the given ID.

Usage:
    python change_id.py <new_id>
"""

import sys
import os
import sysconfig

_sp = os.path.join(sysconfig.get_path("platlib"), "cartesian_hand")
_build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
for _d in (_sp, _build):
    if os.path.isdir(_d):
        sys.path.insert(0, _d)
        break
from ft_servo_ext import FtServo

PORT     = "/dev/ttyACM0"
# PORT     = "/dev/ttyACM1"
START_ID = 0
END_ID   = 20

if len(sys.argv) != 2:
    print("Usage: python change_id.py <new_id>")
    sys.exit(1)

new_id = int(sys.argv[1])

drv = FtServo(PORT)

print(f"Scanning {PORT} for servos (IDs {START_ID}–{END_ID})...")
found = drv.scan(START_ID, END_ID)
print(f"Found: {found}")

if len(found) == 0:
    print("❌ No servos found. Check wiring and port.")
    drv.close()
    sys.exit(1)

if len(found) > 1:
    print(f"❌ Expected exactly 1 servo, but found {len(found)}: {found}")
    print("   Disconnect all but one and retry.")
    drv.close()
    sys.exit(1)

old_id = found[0]
print(f"✔ Found servo ID {old_id}")

if old_id == new_id:
    print(f"Servo is already ID {new_id}. Nothing to do.")
    drv.close()
    sys.exit(0)

drv.write_id(old_id, new_id)
print(f"→ ID changed: {old_id} → {new_id}")

# Verify
drv2 = FtServo(PORT)
ret = drv2.ping(new_id)
if ret >= 0:
    print(f"✔ Verified: servo now responds at ID {new_id}")
else:
    print(f"⚠ Warning: ping at new ID {new_id} failed — verify manually")
drv.close()
drv2.close()