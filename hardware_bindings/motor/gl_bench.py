#!/usr/bin/env python3
"""GL40 II hardware bench test (Position-Velocity / mode 1).

SETUP (once):
  cd /home/grl/repo/DukeHumanoidV2/control
  python gl_setup_can.py                 # brings up can26 @ 1 Mbps
  # power the driver at 24 V, current limit >= 5 A (avoid under-voltage / blinking red)
  # SECURE the motor before any 'move'/'sweep' stage

RUN  (from the control/ directory):
  python hardware_bindings/motor/gl_bench.py read     # enable, read feedback, disable  (no commanded motion)
  python hardware_bindings/motor/gl_bench.py move     # small position move +-0.3 rad @ 1 rad/s
  python hardware_bindings/motor/gl_bench.py sweep    # continuous +-pi/12 sweep via RT thread (sudo for RT prio)
  python hardware_bindings/motor/gl_bench.py zero     # set current shaft position as the new home (redefines 0)

Tips:
  - sudo python hardware_bindings/motor/gl_bench.py sweep   # enables SCHED_FIFO RT priority (optional)
  - watch raw traffic in another terminal:  candump -t z can26
"""
import sys, time, math

from hardware_bindings.motor.py_gl_motor import GLMotorController, GL40, GL_MODE_POS_VEL

CAN, NODE = "can26", 0x01
stage = sys.argv[1] if len(sys.argv) > 1 else "read"


def banner(s): print(f"\n{'='*60}\n{s}\n{'='*60}")


# Construct: py wrapper defaults to POS_VEL; ctor disables + clears faults.
banner(f"GLMotorController([[0x{NODE:02x}, '{CAN}', GL40]], POS_VEL)   stage={stage}")
m = GLMotorController([[NODE, CAN, GL40]], control_mode=GL_MODE_POS_VEL, default_vel_limit=1.0)
m.should_print_send = False
m.should_print_recv = False
time.sleep(0.2)


def show(tag=""):
    p, v, t = float(m.mech_pos[0]), float(m.mech_vel[0]), float(m.mech_torque[0])
    st, er = int(m.mode_status[0]), int(m.error_code[0])
    print(f"  {tag:14s} pos={p:+.3f} rad  vel={v:+.2f}  torq={t:+.2f}  status={st} err={er}")


if stage == "read":
    banner("enable -> read feedback -> disable (no commanded motion)")
    m.enable(); time.sleep(0.2); show("enabled")
    for _ in range(10):
        m.motion_control_once(); time.sleep(0.05)   # holds current pos (vel_ref=limit)
    show("after hold")
    m.disable(); time.sleep(0.1); show("disabled")

elif stage == "move":
    banner("enable -> move +-0.3 rad @ 1 rad/s -> disable  (SECURE MOTOR)")
    m.enable(); time.sleep(0.2)
    start = float(m.mech_pos[0]); show("start")
    m.mech_vel_ref[0] = 1.0   # velocity limit [rad/s]

    def go(target, secs):
        m.mech_pos_ref[0] = target
        t0 = time.time()
        while time.time() - t0 < secs:
            m.motion_control_once(); time.sleep(0.02)
        show(f"cmd={target:+.3f}")

    go(start, 0.6)
    go(start - 0.3, 1.2)
    go(start, 1.2)
    m.disable(); show("disabled")

elif stage == "sweep":
    banner("enable -> continuous +-pi/12 sweep @ 0.5 Hz -> disable  (SECURE MOTOR)")
    m.enable(); time.sleep(0.2)
    start = float(m.mech_pos[0]); show("start")
    m.mech_vel_ref[0] = 3.0   # velocity limit [rad/s]
    m.start_motion_control_continuously()
    amp = math.pi / 12
    t0 = time.time()
    try:
        while time.time() - t0 < 6.0:
            t = time.time() - t0
            m.mech_pos_ref[0] = start + amp * math.sin(2 * math.pi * 0.5 * t)
            time.sleep(0.02)
            print(f"\r  t={t:4.1f}s ref={m.mech_pos_ref[0]:+.3f} pos={m.mech_pos[0]:+.3f} "
                  f"vel={m.mech_vel[0]:+.2f}", end="", flush=True)
    finally:
        print()
        m.mech_pos_ref[0] = start
        time.sleep(0.3)
        m.stop(); time.sleep(0.2)
        m.disable()
        lat = m.get_motor_latency_history(0)
        if lat:
            import statistics
            print(f"  latency n={len(lat)} mean={statistics.mean(lat)*1e3:.2f}ms max={max(lat)*1e3:.2f}ms")

elif stage == "zero":
    banner("set current shaft position as new HOME (redefines 0)")
    m.enable(); time.sleep(0.2); show("before")
    m.set_pos_zero(); time.sleep(0.3); show("after zero")
    m.disable()

else:
    print(f"unknown stage: {stage}  (use: read | move | sweep | zero)")
    m.shutdown(); sys.exit(1)

banner("shutdown")
m.shutdown()
print("done.")
