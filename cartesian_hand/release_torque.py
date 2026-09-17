"""Emergency stop: cut torque on every servo on the bus so the hand can be
moved by hand. `/dev/ttyACM0` may need to be a `/dev/serial/by-id/...` path
instead -- see the port comment on `HAND_2`/`HAND_3` in config.py."""
from cartesian_hand.servo import open_driver

bus = open_driver('/dev/ttyACM0')
ids = bus.scan(0, 32)
print('found', ids)
bus.enable_torques(ids, False)
bus.close()
print('torque released')