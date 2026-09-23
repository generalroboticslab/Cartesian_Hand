"""
FtServo — standalone driver for FeeeTech HLS-series servos (e.g. HLS3915).

Protocol: SCS (half-duplex UART, 0xFF 0xFF header, 8N1).
Tested at 1 Mbps on HLS3915.

Register map (HLS) vs SMS/STS differences:
  - Reg 44-45: GOAL_TORQUE  (SMS/STS has GOAL_TIME)
  - Position write block starts at HLS_ACC (41), 7 bytes:
      [acc, pos_l, pos_h, torq_l, torq_h, spd_l, spd_h]
    SMS/STS writes 6 bytes from GOAL_POSITION_L (42):
      [pos_l, pos_h, time_l, time_h, spd_l, spd_h]
  - Signed values use sign-magnitude (bit 15 = sign) not two's complement.

Interface mirrors StServo (st_servo.py) with one signature change:
  set_position(id, position, speed=0, acc=50, torque=500)
  instead of set_position(id, position, time=0, speed=0)
"""

import serial
import time

# ── Register addresses ────────────────────────────────────────────────────────
HLS_TORQUE_ENABLE      = 40
HLS_ACC                = 41
HLS_GOAL_POSITION_L    = 42
HLS_GOAL_POSITION_H    = 43
HLS_GOAL_TORQUE_L      = 44  # NB: GOAL_TIME on SMS/STS
HLS_GOAL_TORQUE_H      = 45
HLS_GOAL_SPEED_L       = 46
HLS_GOAL_SPEED_H       = 47
HLS_LOCK               = 55

HLS_MIN_ANGLE_LIMIT_L  = 9
HLS_MAX_ANGLE_LIMIT_L  = 11
HLS_OFS_L              = 31
HLS_MODE               = 33

HLS_PRESENT_POSITION_L = 56
HLS_PRESENT_POSITION_H = 57
HLS_PRESENT_SPEED_L    = 58
HLS_PRESENT_SPEED_H    = 59
HLS_PRESENT_LOAD_L     = 60
HLS_PRESENT_LOAD_H     = 61
HLS_PRESENT_VOLTAGE    = 62
HLS_PRESENT_TEMPERATURE = 63
HLS_MOVING             = 66
HLS_PRESENT_CURRENT_L  = 69
HLS_PRESENT_CURRENT_H  = 70

# ── Protocol instruction codes ────────────────────────────────────────────────
INST_PING       = 0x01
INST_READ       = 0x02
INST_WRITE      = 0x03
INST_REG_WRITE  = 0x04
INST_REG_ACTION = 0x05
INST_SYNC_READ  = 0x82
INST_SYNC_WRITE = 0x83


class FtServo:
    """Driver for FeeeTech HLS-series servos over a half-duplex UART bus."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.1):
        self.serial = serial.Serial(port, baudrate, timeout=timeout)

    def close(self):
        self.serial.close()

    # ── Low-level packet I/O ──────────────────────────────────────────────────

    def _calc_checksum(self, id: int, length: int, instruction: int, params) -> int:
        return (~(id + length + instruction + sum(params))) & 0xFF

    def _write_packet(self, id: int, instruction: int, params):
        length = len(params) + 2
        checksum = self._calc_checksum(id, length, instruction, params)
        packet = bytearray(6 + len(params))
        packet[0] = packet[1] = 0xFF
        packet[2] = id
        packet[3] = length
        packet[4] = instruction
        packet[5:5 + len(params)] = params
        packet[5 + len(params)] = checksum
        self.serial.write(packet)

    def _read_packet(self):
        header = self.serial.read(2)
        if len(header) < 2 or header[0] != 0xFF or header[1] != 0xFF:
            return None

        resp_info = self.serial.read(2)
        if len(resp_info) < 2:
            return None
        id, length = resp_info

        payload = self.serial.read(length)
        if len(payload) != length:
            return None

        checksum = payload[-1]
        data_bytes = payload[:-1]  # error byte + params

        calculated = (~sum(list(resp_info) + list(data_bytes))) & 0xFF
        if calculated != checksum:
            return None

        error = data_bytes[0]
        params = data_bytes[1:]
        return id, error, params

    # ── Register access ───────────────────────────────────────────────────────

    def sync_write(self, address: int, data_len: int, data_list):
        """Broadcast sync write. data_list: list of (id, [bytes])."""
        flat = []
        for sid, data in data_list:
            flat.append(sid)
            flat.extend(data)
        self._write_packet(0xFE, INST_SYNC_WRITE, [address, data_len] + flat)

    def write_register(self, id: int, address: int, data, wait_response: bool = True):
        self._write_packet(id, INST_WRITE, [address] + list(data))
        if wait_response:
            return self._read_packet()
        return None

    def read_register(self, id: int, address: int, length: int):
        self._write_packet(id, INST_READ, [address, length])
        ret = self._read_packet()
        if ret is None:
            return None
        _, _, data = ret
        return data

    def read_byte(self, id: int, address: int):
        data = self.read_register(id, address, 1)
        if data and len(data) >= 1:
            return data[0]
        return None

    def read_word(self, id: int, address: int):
        data = self.read_register(id, address, 2)
        if data and len(data) >= 2:
            return data[0] | (data[1] << 8)
        return None

    def read_s16(self, id: int, address: int):
        """Read 16-bit signed value using HLS sign-magnitude encoding (bit 15 = sign)."""
        val = self.read_word(id, address)
        if val is None:
            return None
        if val & 0x8000:
            return -(val & 0x7FFF)
        return val

    # ── Motion control ────────────────────────────────────────────────────────

    def set_position(self, id, position, speed: int = 0, acc: int = 50, torque: int = 500):
        """
        Command position.
        Writes 7 bytes from HLS_ACC (41): [acc, pos_l, pos_h, torq_l, torq_h, spd_l, spd_h].

        Args:
            id:       servo ID (int) or list/tuple of IDs for sync write
            position: target raw counts 0-4095 (int) or list matching id
            speed:    0 = max; 1-32767 = speed limit in counts/s
            acc:      acceleration 0-254
            torque:   torque limit 0-1000 (0.1% units; 1000 = 100%)
        """
        def pack(pos):
            if pos < 0:
                pos = (-pos) | 0x8000  # sign-magnitude encoding
            return [
                acc,
                pos & 0xFF, (pos >> 8) & 0xFF,
                torque & 0xFF, (torque >> 8) & 0xFF,
                speed & 0xFF, (speed >> 8) & 0xFF,
            ]

        if isinstance(id, int):
            return self.write_register(id, HLS_ACC, pack(position))

        # Sync write
        if isinstance(position, (list, tuple)):
            if len(position) != len(id):
                raise ValueError("id and position lengths must match")
            data_list = [(sid, pack(position[i])) for i, sid in enumerate(id)]
        else:
            p = pack(position)
            data_list = [(sid, p) for sid in id]
        self.sync_write(HLS_ACC, 7, data_list)

    def set_speed(self, id, speed, acc: int = 50, torque: int = 500):
        """
        Set running speed (wheel mode).
        Positive = CW, negative = CCW; bit 15 encodes direction.

        Writes the same 7-byte block as set_position (from HLS_ACC=41) with
        position bytes zeroed, matching the HLS WriteSpec reference behavior.
        Writing only GOAL_SPEED_L is insufficient on HLS hardware.

        Args:
            id:     servo ID or list of IDs
            speed:  -32767..32767; sign = direction
            acc:    acceleration 0-254
            torque: torque limit 0-1000 (0.1% units)
        """
        def pack(spd):
            if spd < 0:
                spd = (-spd) | 0x8000
            elif spd > 32767:
                spd = 32767
            return [
                acc,
                0, 0,                               # position = 0 (ignored in wheel mode)
                torque & 0xFF, (torque >> 8) & 0xFF,
                spd & 0xFF, (spd >> 8) & 0xFF,
            ]

        if isinstance(id, int):
            return self.write_register(id, HLS_ACC, pack(speed))

        if isinstance(speed, (list, tuple)):
            data_list = [(sid, pack(speed[i])) for i, sid in enumerate(id)]
        else:
            d = pack(speed)
            data_list = [(sid, d) for sid in id]
        self.sync_write(HLS_ACC, 7, data_list)

    def set_mode(self, id, mode: int):
        """
        Set control mode: 0 = position, 1 = wheel (speed).

        Safety: in wheel mode the servo runs freely. To avoid sudden motion
        when switching back to position mode, the servo is stopped first
        (zero speed + brief settle) before the mode change is sent.
        """
        if isinstance(id, int):
            if mode == 0:
                # Brake before switching away from wheel mode to prevent position overshoot
                self.write_register(id, HLS_GOAL_SPEED_L, [0, 0])
                time.sleep(0.05)
                return self.write_register(id, HLS_MODE, [mode], wait_response=True)
            return self.write_register(id, HLS_MODE, [mode], wait_response=True)
        # Sync write
        if mode == 0:
            for sid in id:
                self.write_register(sid, HLS_GOAL_SPEED_L, [0, 0])
            time.sleep(0.05)
        data_list = [(sid, [mode]) for sid in id]
        self.sync_write(HLS_MODE, 1, data_list)

    def enable_torque(self, id, enable: bool):
        val = 1 if enable else 0
        if isinstance(id, int):
            return self.write_register(id, HLS_TORQUE_ENABLE, [val], wait_response=True)
        data_list = [(sid, [val]) for sid in id]
        self.sync_write(HLS_TORQUE_ENABLE, 1, data_list)

    # ── State reads ───────────────────────────────────────────────────────────

    def get_position(self, id: int):
        return self.read_s16(id, HLS_PRESENT_POSITION_L)

    def get_speed(self, id: int):
        return self.read_s16(id, HLS_PRESENT_SPEED_L)

    def get_load(self, id: int):
        return self.read_s16(id, HLS_PRESENT_LOAD_L)

    def get_voltage(self, id: int):
        """Returns raw byte; unit is 0.1 V."""
        return self.read_byte(id, HLS_PRESENT_VOLTAGE)

    def get_temperature(self, id: int):
        return self.read_byte(id, HLS_PRESENT_TEMPERATURE)

    # ── EPROM / calibration ───────────────────────────────────────────────────

    def unlock_eprom(self, id: int):
        return self.write_register(id, HLS_LOCK, [0])

    def lock_eprom(self, id: int):
        return self.write_register(id, HLS_LOCK, [1])

    def write_id(self, id: int, new_id: int):
        self.unlock_eprom(id)
        ret = self.write_register(id, 5, [new_id])
        self.lock_eprom(new_id)
        return ret

    def set_position_offset(self, id: int, target_pos: int) -> bool:
        """
        Persistently remap current physical position to report as target_pos.
        Writes calculated offset to EPROM (addr 31).

        Offset register: 12-bit sign-magnitude (bit 11 = sign, bits 0-10 = magnitude).
        Valid offset range: ±2047 counts.
        """
        p_curr = self.get_position(id)
        if p_curr is None:
            return False

        raw_offset = self.read_word(id, HLS_OFS_L)
        if raw_offset is None:
            return False

        # Decode current 12-bit sign-magnitude offset
        sign = -1 if (raw_offset & 0x0800) else 1
        o_curr = sign * (raw_offset & 0x07FF)

        p_raw = (p_curr - o_curr) % 4096
        o_new = (target_pos - p_raw) % 4096
        if o_new > 2047:
            o_new -= 4096

        new_magnitude = abs(o_new) & 0x07FF
        new_raw = new_magnitude | (0x0800 if o_new < 0 else 0)

        self.unlock_eprom(id)
        self.write_register(id, HLS_OFS_L, [new_raw & 0xFF, (new_raw >> 8) & 0xFF], wait_response=True)
        self.lock_eprom(id)
        return True

    # ── Bus utilities ─────────────────────────────────────────────────────────

    def ping(self, id: int):
        self._write_packet(id, INST_PING, [])
        return self._read_packet()

    def scan(self, start_id: int = 0, end_id: int = 253):
        """Scan bus for responding servo IDs."""
        found = []
        original_timeout = self.serial.timeout
        self.serial.timeout = 0.004
        self.serial.reset_input_buffer()
        for i in range(start_id, end_id + 1):
            try:
                self._write_packet(i, INST_PING, [])
                if self._read_packet() is not None:
                    found.append(i)
            except Exception:
                pass
        self.serial.timeout = original_timeout
        return found


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():

    # refer to Servo port (FeeeTech) setup
    # PORT = "/dev/ttyACM2"
    # PORT = "/dev/ttyACMservoLeft"
    PORT = "/dev/ttyACMservoRight"

    SERVO_ID = 1

    print(f"Connecting to {PORT}...")
    servo = FtServo(PORT)

    ret = servo.ping(SERVO_ID)
    if ret is None:
        print(f"Ping failed (ID {SERVO_ID})")
        servo.close()
        return
    _, error, _ = ret
    print(f"Ping OK (ID {SERVO_ID}, error=0x{error:02X})")

    pos = servo.get_position(SERVO_ID)
    volt = servo.get_voltage(SERVO_ID)
    temp = servo.get_temperature(SERVO_ID)
    print(f"  Position: {pos}  Voltage: {volt/10.0 if volt else '?'} V  Temp: {temp} C")

    servo.enable_torque(SERVO_ID, True)

    # targets = [
    #         ("left", 0),
    #         ("center", 2048),
    #         ("right", 4095),
    # ]
    targets = [
            # ("left", 500),
            # ("center", 2048),
        # ("calib", 0),
        ("start", 500),
        ("end", 2000),
        # ("start", 500),
    ]
    
    for label, target in targets:
        print(f"Moving {label} ({target})...")
        servo.set_position(SERVO_ID, target, speed=1000, acc=100, torque=1000)
        time.sleep(1.5)
        print(f"  Position: {servo.get_position(SERVO_ID)}")

    servo.enable_torque(SERVO_ID, False)
    servo.close()
    print("Done.")


if __name__ == "__main__":
    main()
