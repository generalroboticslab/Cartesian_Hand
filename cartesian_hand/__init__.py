"""Cartesian Hand: control stack for a 7-DOF FT-servo gripper.

    from cartesian_hand.hand import connect
    with connect("hand_2") as hand:
        hand.set_pos(hand.config.upper)

Import from the module that owns the name. `hands` defines what a hand is,
`hand` drives one, `policy` runs a twin-developed policy against it.
"""
