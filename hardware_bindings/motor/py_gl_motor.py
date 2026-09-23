import numpy as np
import time

from hardware_bindings.motor.py_motor import motor_bindings

GLMotorController_ext = motor_bindings.GLMotorController

# Motor type ids (must match GLMotorType in gl_motor.hpp)
GL40 = 0

# Control modes (must match GLControlMode in gl_motor.hpp).
#   MIT (0):     pos/vel/kp/kd/tff impedance — needs MIT-enabled firmware.
#   POS_VEL (1): cascaded position loop + velocity limit (mech_pos_ref, mech_vel_ref=limit).
#   VEL (2):     pure velocity loop (mech_vel_ref = target velocity).
# The GL40 II tested shipped in POS_VEL and ignores MIT frames.
GL_MODE_MIT = 0
GL_MODE_POS_VEL = 1
GL_MODE_VEL = 2
GL_MODE_NAMES = {GL_MODE_MIT: "MIT", GL_MODE_POS_VEL: "POS_VEL", GL_MODE_VEL: "VEL"}

# GL40 II MIT torque range (TMAX). Matches gl_spec_gl40 in gl_motor.hpp.
GL_MAX_TORQUE = {
    GL40: 10.0,
}

MOTOR_TYPE_NAMES = {
    GL40: "GL40",
}

# Mapping from GLMotorController.mode_status (MotorModeStatus enum) to a label.
MODE_STATUS_NAMES = {0: "Reset", 1: "Calib", 2: "Running", 255: "Unknown"}

# GL II diagnostic fault codes (raw err nibble in error_code).
GL_ERROR_NAMES = {
    0: "Disabled",
    1: "Enabled",
    8: "OverVolt",
    9: "UnderVolt",
    10: "OverCurr",
    11: "MOSOverTemp",
    12: "MotorOverTemp",
    13: "CommLoss",
    14: "Overload",
}


class GLMotorController(GLMotorController_ext):
    def __init__(self, motor_setup, control_mode=GL_MODE_POS_VEL,
                 default_kp=0.0, default_kd=0.1, default_vel_limit=2.0):
        """
        motor_setup = [
            [can_id, can_interface, motor_type],
            ...
        ]
        GL40 runs on a dedicated CAN bus (never mixed with Robstride).

        control_mode (GLControlMode): POS_VEL (default, matches tested hardware), MIT, or VEL.
          - POS_VEL: command mech_pos_ref [rad]; mech_vel_ref is the speed LIMIT [rad/s].
          - MIT:     command mech_pos_ref/mech_vel_ref/mech_torque_ref with kp/kd gains.
          - VEL:     command mech_vel_ref [rad/s].
        default_vel_limit: initial mech_vel_ref for POS_VEL so position commands actually move.
        """
        can_interfaces = sorted(list(set(e[1] for e in motor_setup)))
        can_interface_id_map = {can_interface: i for i, can_interface in enumerate(can_interfaces)}
        can_interface_ids = np.array([can_interface_id_map[e[1]] for e in motor_setup], dtype=np.uint8)
        can_ids = np.array([e[0] for e in motor_setup], dtype=np.uint32)
        motor_types = np.array([e[2] for e in motor_setup], dtype=np.uint8)

        super().__init__(
            can_interfaces,
            can_interface_ids,
            can_ids,
            motor_types,
            control_mode,
        )
        self.should_print_send = False
        self.should_print_recv = False

        # Start disabled and faults cleared.
        self.disable(True)
        self.disable(False)

        # Default MIT impedance gains (unused in POS_VEL/VEL). Conservative — caller overrides.
        self.kp[:] = default_kp
        self.kd[:] = default_kd
        self.mech_pos_ref[:] = 0.0
        self.mech_torque_ref[:] = 0.0
        # In POS_VEL, mech_vel_ref is the velocity limit; seed it so pos commands move.
        self.mech_vel_ref[:] = default_vel_limit if control_mode == GL_MODE_POS_VEL else 0.0

    def enable(self, motor_enable_mask=None):
        """Enable, then seed mech_pos_ref to the measured position.

        SAFETY: after enabling, the first motion_control_once() sends mech_pos_ref. If it
        were left at its stale value (0 from construction), the motor would jump from its
        current pose to 0 rad. Seeding ref = current pos makes the first command a hold.
        """
        if motor_enable_mask is None:
            super().enable()
        else:
            super().enable(motor_enable_mask)
        time.sleep(0.05)  # let one feedback frame land so mech_pos is current
        self.mech_pos_ref[:] = self.mech_pos[:]

    def shutdown(self):
        self.kp[:] = 0.0
        self.kd[:] = 0.0
        self.mech_torque_ref[:] = 0.0
        self.disable(True)
        self.disable(False)

    def self_check(self):
        """Display GL motor status: type, position, velocity, torque, temps, mode/err."""
        print(f"control_mode: {GL_MODE_NAMES.get(int(self.control_mode), int(self.control_mode))}")
        type_names = [MOTOR_TYPE_NAMES.get(t, f"?{t}") for t in self.motor_types]
        status_names = [MODE_STATUS_NAMES.get(int(s), f"?{int(s)}") for s in self.mode_status]
        err_names = [GL_ERROR_NAMES.get(int(e), f"?{int(e)}") for e in self.error_code]

        # (name, data, fmt)
        cols = [
            ("index",      range(self.num_motors),  ">5d"),
            ("type",       type_names,              ">5s"),
            ("mech_pos",   self.mech_pos,           ">9.3f"),
            ("mech_vel",   self.mech_vel,           ">9.2f"),
            ("mech_torq",  self.mech_torque,        ">9.2f"),
            ("drive_T",    self.drive_temp,         ">7.0f"),
            ("motor_T",    self.motor_temp,         ">7.0f"),
            ("status",     status_names,            ">8s"),
            ("err",        err_names,               ">13s"),
        ]

        def get_width(fmt):
            import re
            return int(re.search(r"\d+", fmt).group())

        header = " | ".join(f"{n:>{get_width(f)}}" for n, _, f in cols)
        lines = [header, "-" * len(header)]
        for i in range(self.num_motors):
            lines.append(" | ".join(f"{d[i]:{f}}" for _, d, f in cols))
        print("\n".join(lines))


class FakeGLMotorController:
    """Drop-in stand-in for GLMotorController on machines without the GL CAN bus."""

    def __init__(self, motor_setup, control_mode=GL_MODE_POS_VEL,
                 default_kp=0.0, default_kd=0.1, default_vel_limit=2.0):
        can_interfaces = sorted(list(set(e[1] for e in motor_setup)))
        can_interface_id_map = {can_interface: i for i, can_interface in enumerate(can_interfaces)}
        self.control_mode = control_mode
        self.can_interface_ids = np.array([can_interface_id_map[e[1]] for e in motor_setup], dtype=np.uint8)
        self.can_ids = np.array([e[0] for e in motor_setup], dtype=np.uint32)
        self.motor_types = np.array([e[2] for e in motor_setup], dtype=np.uint8)
        self.num_motors = len(self.motor_types)

        n = self.num_motors
        self.mech_pos_ref = np.zeros(n, dtype=np.float32)
        self.mech_vel_ref = np.zeros(n, dtype=np.float32)
        self.mech_torque_ref = np.zeros(n, dtype=np.float32)
        self.kp = np.full(n, default_kp, dtype=np.float32)
        self.kd = np.full(n, default_kd, dtype=np.float32)
        self.mech_pos = np.zeros(n, dtype=np.float32)
        self.mech_vel = np.zeros(n, dtype=np.float32)
        self.mech_torque = np.zeros(n, dtype=np.float32)
        self.drive_temp = np.zeros(n, dtype=np.float32)
        self.motor_temp = np.zeros(n, dtype=np.float32)
        self.mode_status = np.zeros(n, dtype=np.uint8)
        self.error_code = np.zeros(n, dtype=np.uint8)

    def shutdown(self):
        print("FakeGLMotorController shutdown")

    def enable(self, motor_enable_mask=None):
        print("FakeGLMotorController enable")

    def disable(self, clear_fault=False):
        print(f"FakeGLMotorController disable clear_fault={clear_fault}")

    def set_pos_zero(self):
        print("FakeGLMotorController set_pos_zero")

    def clear_error(self):
        print("FakeGLMotorController clear_error")

    def motion_control_once(self):
        pass

    def start_motion_control_continuously(self):
        print("FakeGLMotorController start_motion_control_continuously")

    def stop(self):
        print("FakeGLMotorController stop")

    def self_check(self):
        print("FakeGLMotorController self_check")


# ----------------------------------------------------------------------------
# Offline packing reference + test (verification step 3). Mirrors the exact
# truncation packing in gl_motor.cpp; validates against the manual's hex examples
# WITHOUT touching hardware.
# ----------------------------------------------------------------------------
# Default GL40 scaling (gl_spec_gl40). VMAX must be confirmed on hardware.
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -200.0, 200.0
T_MIN, T_MAX = -10.0, 10.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0


def _float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max
    return int((x - x_min) * ((1 << bits) - 1) / span)  # truncation (matches C (int) cast)


def pack_mit_cmd(p_des, v_des, kp, kd, t_ff,
                 p_limits=(P_MIN, P_MAX), v_limits=(V_MIN, V_MAX),
                 t_limits=(T_MIN, T_MAX), kp_limits=(KP_MIN, KP_MAX),
                 kd_limits=(KD_MIN, KD_MAX)):
    p_int = _float_to_uint(p_des, *p_limits, 16)
    v_int = _float_to_uint(v_des, *v_limits, 12)
    kp_int = _float_to_uint(kp, *kp_limits, 12)
    kd_int = _float_to_uint(kd, *kd_limits, 12)
    t_int = _float_to_uint(t_ff, *t_limits, 12)
    data = bytearray(8)
    data[0] = (p_int >> 8) & 0xFF
    data[1] = p_int & 0xFF
    data[2] = (v_int >> 4) & 0xFF
    data[3] = (((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)) & 0xFF
    data[4] = kp_int & 0xFF
    data[5] = (kd_int >> 4) & 0xFF
    data[6] = (((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)) & 0xFF
    data[7] = t_int & 0xFF
    return bytes(data)


def test_gl_packing():
    # MIT position: p=+2.0, kp=0.123, kd=0.005 -> 94 7A 7F F0 01 00 47 FF
    got = pack_mit_cmd(2.0, 0.0, 0.123, 0.005, 0.0)
    assert got == bytes.fromhex("947A7FF0010047FF"), got.hex()

    # MIT position: p=-2.0, kp=0.123, kd=0.005 -> 6B 84 7F F0 01 00 47 FF
    got = pack_mit_cmd(-2.0, 0.0, 0.123, 0.005, 0.0)
    assert got == bytes.fromhex("6B847FF0010047FF"), got.hex()

    # MIT speed example uses the ±250 scale (manual discrepancy). Verify with that scale:
    # v=+6.0, kd=0.005 -> 7F FF 83 00 00 00 47 FF
    got = pack_mit_cmd(0.0, 6.0, 0.0, 0.005, 0.0, v_limits=(-250.0, 250.0))
    assert got == bytes.fromhex("7FFF83000000 47FF".replace(" ", "")), got.hex()

    print("test_gl_packing PASSED")


if __name__ == "__main__":
    test_gl_packing()
