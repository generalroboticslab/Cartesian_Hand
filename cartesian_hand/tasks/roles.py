"""Names for the DOF indices of the standard 7-DOF layout.

Tasks address DOFs by mechanical role. `hand.set_dofs({AUX_JAW: 12.0})` says
what it does; `set_pos([None, None, None, None, 12.0, None, None])` does not.

These match hands.LAYOUT. A hand with a different layout needs its own role map.
"""

BASE_JAW = 0        # base parallel actuation, y
BASE_LEFT = 1       # base left finger, x
BASE_RIGHT = 2      # base right finger, x
Z = 3               # vertical translation
AUX_JAW = 4         # aux parallel actuation, y
AUX_LEFT = 5        # aux left finger, x
AUX_RIGHT = 6       # aux right finger, x

JAWS = [BASE_JAW, AUX_JAW]
BASE_FINGERS = [BASE_LEFT, BASE_RIGHT]
AUX_FINGERS = [AUX_LEFT, AUX_RIGHT]
