"""Cartesian Hand: control stack for a 7-DOF FT-servo gripper.

    from cartesian_hand.hand import connect
    with connect("hand_2") as hand:        # requires zeroing first
        hand.set_pos(hand.config.upper)

Import from the module that owns the name. `hand` defines what a hand is and
drives it; `policy` runs a twin-developed policy against it.
"""
