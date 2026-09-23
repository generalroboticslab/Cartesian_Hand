import gc
import importlib.util
import pathlib
from enum import Enum
from functools import lru_cache

import numpy as np
import time


@lru_cache(maxsize=None)
def _load_ext(name):
    path = pathlib.Path(__file__).parent / f"{name}.abi3.so"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run `cmake --build control/build` first")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




motor_bindings = _load_ext("motor_bindings")
MotorParams = motor_bindings.MotorParams

# from common.publisher import DataPublisher

R00 = 0
R01 = 1
R02 = 2
R03 = 3
R04 = 4
R05 = 5
R06 = 6


MAX_TORQUE = {
    R01:17.0,
    R02:17.0,
    R03:60.0,
    R04:120.0,
    R05:5.5,
    R06:36,
    R00:14,
}

MOTOR_CURRENT_LIMIT_DEFAULTS = { #Arms
    R00: 16.0,  
    R01: 23.0,
    R02: 23.0,
    R03: 43.0,
    R04: 60.0,  # 90
    R05: 11.0,  
    R06: 57.0,
}

MOTOR_TORQUE_CONSTANTS = { # Nm/Arms
    R00: 1.48,  
    R01: 1.22,
    R02: 1.22,
    R03: 2.36,
    R04: 2.10, 
    R05: 0.94,
    R06: 1.1, 
}

MOTOR_RESISTANCE = { # Ohm +- 10%
    R00: 1.5,
    R01: 0.55,
    R02: 0.55,
    R03: 0.39,
    R04: 0.16,
    R05: 2.72,
    R06: 0.23,
}

MOTOR_BACK_EMF_CONSTANTS = { # Vrms/(rad/s)
    R00: 0.91,   # 0.095 Vrms/rpm
    R01: 0.92,   # 0.096 Vrms/rpm
    R02: 0.92,   # 0.096 Vrms/rpm
    R03: 0.16,   # 17.0 Vrms/krpm
    R04: 0.16,   # 16.9 Vrms/krpm
    R05: 0.071,  # 7.4 Vrms/krpm
    R06: 0.073,  # 7.6 Vrms/krpm
}

# R03 / R04 / R06 → cur_kp = 0.17, cur_ki = 0.01
# R02 / R00 / R05 → cur_kp = 0.12, cur_ki = 0.02
MOTOR_CUR_KP = {
    R00: 0.12,
    R01: 0.12,
    R02: 0.12,
    R03: 0.17,
    R04: 0.17,
    R05: 0.12,
    R06: 0.17,
}

MOTOR_CUR_KI = {
    R00: 0.016,
    R01: 0.016,
    R02: 0.016,
    R03: 0.012,
    R04: 0.012,
    R05: 0.016,
    R06: 0.012,
}


MOTOR_TYPE_NAMES = {
    R00: "R00", R01: "R01", R02: "R02", R03: "R03",
    R04: "R04", R05: "R05", R06: "R06"
}



class MOTO_PARAM(Enum):
    run_mode = 0X7005      # 0X7005 0: Operation control mode 1: Position mode 2: Speed mode 3: Current mode 4: Zero mode uint8  1byte
    iq_ref = 0X7006        # 0X7006 Current mode Iq command               W/R  float   4byte   -23~23A
    mech_vel_ref = 0X700A  # 0x700A Speed mode speed command              W/R  float   4byte   -30~30rad/s
    torque_limit = 0X700B  # 0x700B Torque limit                          W/R  float   4byte   {0~12Nm}{0~60Nm}{0~120Nm}
    cur_kp = 0X7010        # 0x7010 Current Kp                            W/R  float   4byte   Default value 0.125
    cur_ki = 0X7011        # 0x7011 Current Ki                            W/R  float   4byte   Default value 0.0158
    cur_filt_gain = 0X7014 # 0x7014 Current filter coefficient filt_gain  W/R  float   4byte   0~1.0, default value 0.1
    mech_pos_ref = 0X7016  # 0x7016 Position mode angle command           W/R  float   4byte   rad
    spd_limit = 0X7017     # 0x7017 Position mode speed limit             W/R  float   4byte   0~30rad/s
    cur_limit = 0X7018     # 0x7018 Current mode current limit            W/R  float   4byte   0~23A
    mech_pos = 0X7019      # 0x7019 Load end count mechanical angle         R  float   4byte   rad
    iqf = 0X701A           # 0x701A iq filter value                         R  float   4byte   -23~23A
    mech_vel = 0X701B      # 0x701B Load end count speed                    R  float   4byte   -30~30rad/s
    v_bus = 0X701C         # 0x701C Bus voltage                             R  float   4byte   V
                            # rotation = 0X701D,      // 0x701D Number of turns                         R  int16   2byte   Number of turns (only for Robstrid01)
    loc_kp = 0X701E        # 0x701E position kp                           W/R  float   4byte   Default value 30
    spd_kp = 0X701F        # 0x701F speed kp                              W/R  float   4byte   Default value 1
    spd_ki = 0X7020        # 0x7020 speed ki                              W/R  float   4byte   Default value 0.002
    spd_filt_gain = 0X7021 # 0x7021 speed filter coefficient filt_gain    W/R  float   4byte   0~1.0, default value 0.1
    zero_sta = 0X7029      # 0x7029 Zero point flag                       W/R  uint8   1byte   0: 0–2π (default), 1: -π–+π
    CAN_TIMEOUT=0X200b     # 0X200b CAN timeout 20000 = 1s                W/R  uint32  4byte   0-100000


class CanMotorController(motor_bindings.CanMotorController):
    def __init__(self, motor_setup):
        """
        motor_setup = [
            [can_id, can_interface, motor_type],
            [can_id, can_interface, motor_type],
            [can_id, can_interface, motor_type],
            [can_id, can_interface, motor_type],
        ]
        
        """
        can_interfaces = sorted(list(set(e[1] for e in motor_setup)))
        can_interface_id_map = {can_interface: i for i, can_interface in enumerate(can_interfaces)}
        can_interface_ids = np.array([can_interface_id_map[e[1]] for e in motor_setup], dtype=np.uint8)
        can_ids = np.array([e[0] for e in motor_setup], dtype=np.uint32)
        motor_types = np.array([e[2] for e in motor_setup], dtype=np.uint8)

        # print(f"can_interfaces: {can_interfaces}")
        # print(f"can_interface_ids: {can_interface_ids}")
        # print(f"can_ids: {can_ids}")
        # print(f"motor_types: {motor_types}")


        ## example of the setup to the parameters
        # can_interfaces = ["can6","can7","can8","can24"]
        # can_interface_ids = np.array([
        # 3,3,3,3,3,
        # 2,0,1,1,1,
        # 2,0,1,1,2,
        # 2,0,0,0,2], dtype=np.uint8)
        # can_ids = np.array([0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10,0x11,0x12,0x13,0x14], dtype=np.uint32)
        # motor_types = np.array([R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02,R02], dtype=np.uint8)

        super().__init__(
            can_interfaces,
            can_interface_ids,
            can_ids,
            motor_types
        )
        self.should_print_send = False
        self.should_print_recv = False
        self.disable(True)
        self.disable(False)

        self.getAllParams()
        # self.should_print_send = True
        # self.should_print_recv = True
        # print(f"{__file__}: Hack todo change back")

        # cantimeout: units are 20000 = 1s per motor.hpp comment, but verify on hardware.
        # 5000 was a hack to survive startup — now fixed via recv_timeout_ms.
        # TODO: reduce once units are confirmed on hardware (target: ~100ms = 2000 units).
        self.setParam(0x7028,np.full(len(can_ids),5000,dtype=np.uint32))
        # time.sleep(0.1)
        # self.getParam(0X200b)
        # self.getParam(0X7028)
        # self.should_print_send = False
        # self.should_print_recv = False

        # # set spd_filt to 0.05 # larger-> more up to date but more noisy
        # self.setParam(MOTO_PARAM.spd_filt_gain.value, np.full(len(can_ids),0.1,dtype=np.float32))
        # # set cur_filt # larger-> more up to date but more noisy
        # self.setParam(MOTO_PARAM.cur_filt_gain.value, np.full(len(can_ids),0.1,dtype=np.float32))

        # # spd_ki # no effect in operation control mode
        # self.setParam(MOTO_PARAM.spd_ki.value, np.full(len(can_ids),0.016,dtype=np.float32))

        # # set higher cur_kp than default
        # cur_kp = np.array([MOTOR_CUR_KP[t] for t in self.motor_types], dtype=np.float32) * 1.1 # HACK increase kp by 10%
        # self.setParam(MOTO_PARAM.cur_kp.value, cur_kp)

        # # set cur_ki
        # cur_ki = np.array([MOTOR_CUR_KI[t] for t in self.motor_types], dtype=np.float32)
        # self.setParam(MOTO_PARAM.cur_ki.value, cur_ki)


        self.max_motor_torques = np.array([MAX_TORQUE[k] for k in motor_types],dtype=np.float32)
        print("default max torque to 0.05")
        self.set_max_torque_ratio(0.05)
        
        # print(self)
        # print(f"can_ids: {self.can_ids}")
        # print(f"motor_types: {self.motor_types}")
        # print(self.can_ids.dtype)
        # print(type(self.can_ids))

    def set_max_torque_ratio(self,ratio:float):
        """set max torqu ratio between [0,1]"""
        if ratio > 1 or ratio <= 0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.setParam(MOTO_PARAM.torque_limit.value,self.max_motor_torques*ratio)

    def shutdown(self):
        self.loc_kp[:] = 0
        self.spd_kp[:] = 0
        self.set_max_torque_ratio(0.001)
        # time.sleep(0.05)
        self.disable(True)
        self.disable(False)

    def self_check(self):
        """Display motor status: type, bus voltage, position, torque limit, current limit""" 
        self.set_max_torque_ratio(0.8)
        time.sleep(0.02)
       
        self.getAllParams()
        avg_v_bus = np.mean(self.v_bus)
        type_names = [MOTOR_TYPE_NAMES.get(t, f"?{t}") for t in self.motor_types]


        kt = np.array([MOTOR_TORQUE_CONSTANTS[t] for t in self.motor_types], dtype=np.float32)
        ke = np.array([MOTOR_BACK_EMF_CONSTANTS[t] for t in self.motor_types], dtype=np.float32)
        r = np.array([MOTOR_RESISTANCE[t] for t in self.motor_types], dtype=np.float32)
        max_torque_est = self.cur_limit * kt

        max_vel_at_max_torque = (avg_v_bus - self.torque_limit/kt*r)/ke
        
        # (name, data, fmt)
        cols = [
            ("index",    range(len(self.v_bus)),  ">5d"),
            ("type",     type_names,              ">4s"),
            ("v_bus",    self.v_bus,              ">7.2f"),
            ("v_diff",   self.v_bus - avg_v_bus,  ">+7.2f"),
            ("mech_pos", self.mech_pos,           ">8.2f"),
            ("cur_lim",  self.cur_limit,          ">7.1f"),
            ("est_torq", max_torque_est,          ">8.1f"),
            ("est_vel",  max_vel_at_max_torque,   ">8.1f"),
            ("torq_lim", self.torque_limit,       ">8.1f"),
            ("loc_kp",   self.loc_kp,             ">6.1f"),
            ("spd_kp",   self.spd_kp,             ">6.2f"),
            ("spd_ki",   self.spd_ki,             ">6.4f"),
            ("cur_kp",   self.cur_kp,             ">6.3f"),
            ("cur_ki",   self.cur_ki,             ">6.3f"),
            ("cur_filt", self.cur_filt_gain,      ">6.2f"),
            ("spd_filt", self.spd_filt_gain,      ">8.2f"),
            ("can_to",   self.can_timeout,        ">6d"),
        ]
        self.set_max_torque_ratio(0.05)

        def get_width(fmt):
            import re
            return int(re.search(r'\d+', fmt).group())

        header = " | ".join(f"{n:>{get_width(f)}}" for n, _, f in cols)
        lines = [f"V_bus average: {avg_v_bus:.3f}V", header, "-" * len(header)]
        for i in range(len(self.v_bus)):
            lines.append(" | ".join(f"{d[i]:{f}}" for _, d, f in cols))
        print("\n".join(lines))
            
class FakeMotorController:
    def __init__(self, motor_setup):
        can_interfaces = sorted(list(set(e[1] for e in motor_setup)))
        can_interface_id_map = {can_interface: i for i, can_interface in enumerate(can_interfaces)}
        can_interface_ids = np.array([can_interface_id_map[e[1]] for e in motor_setup], dtype=np.uint8)
        can_ids = np.array([e[0] for e in motor_setup], dtype=np.uint32)
        self.motor_types = np.array([e[2] for e in motor_setup], dtype=np.uint8)
        self.max_motor_torques = np.array([MAX_TORQUE[k] for k in self.motor_types],dtype=np.float32)
        
        self.torque_limit = np.zeros(len(self.motor_types),dtype=np.float32)
        self.loc_kp = np.zeros(len(self.motor_types),dtype=np.float32)
        self.spd_kp = np.zeros(len(self.motor_types),dtype=np.float32)
        self.mech_pos = np.zeros(len(self.motor_types),dtype=np.float32)
        self.mech_vel = np.zeros(len(self.motor_types),dtype=np.float32)
        self.mech_pos_ref = np.zeros(len(self.motor_types),dtype=np.float32)
        self.mech_torque = np.zeros(len(self.motor_types),dtype=np.float32)
        self.temperature = np.zeros(len(self.motor_types),dtype=np.float32)
        self.mech_torque_ref = np.zeros(len(self.motor_types),dtype=np.float32)

    
    def set_max_torque_ratio(self,ratio:float):
        self.torque_limit[:] = self.max_motor_torques*ratio


    def shutdown(self):
        print("FakeMotorController shutdown")

    def enable(self, motor_enable_mask=None):
        print("FakeMotorController enable")

    def disable(self):
        print("FakeMotorController disable")

    def getAllParams(self):
        print("FakeMotorController getAllParams")

    def setParam(self,param,value):
        print(f"FakeMotorController setParam {param} {value}")

    def getParam(self,param):
        print(f"FakeMotorController getParam {param}")

    def start_motion_control_continuously(self):
        print("FakeMotorController start_motion_control_continuously")


def test_motor_nanobind():
    assert MotorParams.run_mode == 0X7005
    assert MotorParams.iq_ref == 0X7006
    assert MotorParams.mech_vel_ref == 0X700A
    assert MotorParams.torque_limit == 0X700B
    assert MotorParams.cur_kp == 0X7010
    assert MotorParams.cur_ki == 0X7011
    assert MotorParams.cur_filt_gain == 0X7014
    assert MotorParams.mech_pos_ref == 0X7016
    assert MotorParams.spd_limit == 0X7017
    assert MotorParams.cur_limit == 0X7018
    assert MotorParams.mech_pos == 0X7019
    assert MotorParams.iqf == 0X701A
    assert MotorParams.mech_vel == 0X701B
    assert MotorParams.v_bus == 0X701C
    # assert MotorParams.rotation == 0X701D
    assert MotorParams.loc_kp == 0X701E
    assert MotorParams.spd_kp == 0X701F
    assert MotorParams.spd_ki == 0X7020
    assert MotorParams.spd_filt_gain == 0X7021

if __name__ == "__main__":
    test_motor_nanobind()
