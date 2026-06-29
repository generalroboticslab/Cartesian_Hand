"""
scan.py — scan bus for available servo IDs.

Usage:
    python commons/scan.py [port] [start_id] [end_id]

    port    : serial port (default: /dev/ttyACM1)
    start_id: first ID to scan (default: 0)
    end_id  : last ID to scan (default: 20)
"""

import sys
import os
from cartesian_hand.ft_servo_ext import FtServo

port     =  "/dev/ttyACM0"
# port     =  "/dev/ttyACM1"
start_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end_id   = int(sys.argv[3]) if len(sys.argv) > 3 else 20

drv = FtServo(port)
print(f"Scanning {port} IDs {start_id}-{end_id}...")
found = drv.scan(start_id, end_id)
print(f"Found: {found}")

for sid in found:
    pos  = drv.get_position(sid)
    volt = drv.get_voltage(sid)
    temp = drv.get_temperature(sid)
    print(f"  ID {sid}: pos={pos}  volt={volt/10:.1f}V  temp={temp}C")

drv.close()
