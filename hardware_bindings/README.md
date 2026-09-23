# hardware_bindings

Generic C++ nanobind extensions for motor control, IMU, and FT servo hardware.

## Modules

- `motor_bindings` — CAN motor controller (DukeHumanoidV2 leg/arm motor primitives)
- `imu_nanobind` — IMU serial reader (TM3xx via EasyProfile protocol)
- `ft_servo_ext` — Feetech SCServo / HLSCL serial driver. See [`ft_servo/README.md`](ft_servo/README.md)
  for the driver's read/write surface and its `scan` / `set-id` / `gui` bench tools.

## Install

```bash
pip install -e .
```

Build deps required on system: Eigen3, CSerialPort (itas109), fmt, nanobind, Python 3.12+.

## Import

```python
from hardware_bindings import motor_bindings, imu_nanobind, ft_servo_ext
```

## Submodule usage

In `DukeHumanoidV2`:

```bash
git submodule update --init --recursive
pip install -e control/hardware_bindings
```