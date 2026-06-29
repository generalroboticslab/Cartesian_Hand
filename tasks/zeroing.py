import argparse
import sys
import os
import time
import json
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartesian_hand import config_1

DEFAULT_OFFSETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "commons", "zero_offsets.json"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config_key(controller) -> str:
    return "config_1" if controller.dof_config is config_1 else "config_2"


# ── Persistence ───────────────────────────────────────────────────────────────

def save_offsets(controller, path: str = DEFAULT_OFFSETS_PATH):
    """Save this controller's offsets into the shared JSON (merges with the other config)."""
    key  = _config_key(controller)
    data = {"config_1": None, "config_2": None}
    if os.path.exists(path):
        with open(path) as f:
            data.update(json.load(f))
    data[key]         = controller.zero_offset.tolist()
    data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Zero offsets saved [{key}] → {path}")


def load_offsets(controller, path: str = DEFAULT_OFFSETS_PATH) -> bool:
    """Load this controller's offsets from the shared JSON. Returns False if missing/null."""
    if not os.path.exists(path):
        print(f"No saved offsets at {path}")
        return False
    key = _config_key(controller)
    try:
        with open(path) as f:
            data = json.load(f)
        offsets = data.get(key)
        if offsets is None:
            print(f"No saved offsets for {key}")
            return False
        if len(offsets) != 7:
            print(f"Bad offset data for {key}: expected 7 values, got {len(offsets)}")
            return False
        controller.zero_offset[:] = offsets
        controller.is_zeroed = True
        print(f"Loaded [{key}] offsets from {path}  [{data.get('timestamp', '?')}]")
        print(f"  zero_offset: {controller.zero_offset.tolist()}")
        return True
    except Exception as e:
        print(f"Failed to load offsets: {e}")
        return False


# ── Zeroing routines ──────────────────────────────────────────────────────────

def zero_all(controller, stall_threshold: int = 5, confirm_count: int = 2,
             zero_torque: int = 50, creep_speed: int = 50,
             transit_speed: int = 100, transit_acc: int = 20,
             transit_torque: int = 400, transit_tolerance: int = 80,
             transit_timeout: float = 5.0,
             save: bool = True, save_path: str = DEFAULT_OFFSETS_PATH):
    """Drive all DOFs to hard stop in phase order, then move to halfway.
    Phase 1: x-slides (dof 1,2,5,6)
    Phase 2: y-jaws   (dof 0,4)
    Phase 3: z-axis   (dof 3)
    """
    cfg = controller.dof_config

    DOF_PHASE1 = [1, 2, 5, 6]
    DOF_PHASE2 = [0, 4]
    DOF_PHASE3 = [3]

    def run_phase(dof_ids):
        zeroed      = {d: False for d in dof_ids}
        consecutive = {d: 0     for d in dof_ids}
        prev_actual = {d: controller.servo.read_position(cfg[d]["servo_id"]) for d in dof_ids}

        for dof_id in dof_ids:
            sid         = cfg[dof_id]["servo_id"]
            orientation = cfg[dof_id]["orientation"]
            far_target  = prev_actual[dof_id] - int(orientation * 10000)
            controller.servo.set_position(sid, far_target, creep_speed, 20, zero_torque)

        while not all(zeroed.values()):
            time.sleep(0.1)
            for dof_id in dof_ids:
                if zeroed[dof_id]:
                    continue
                sid    = cfg[dof_id]["servo_id"]
                actual = controller.servo.read_position(sid)
                movement = (abs(actual - prev_actual[dof_id])
                            if actual is not None and prev_actual[dof_id] is not None else 999)
                print(f"  DOF {dof_id} | actual: {actual} | movement: {movement}")
                prev_actual[dof_id] = actual
                if movement < stall_threshold:
                    consecutive[dof_id] += 1
                    if consecutive[dof_id] >= confirm_count:
                        controller.zero_offset[dof_id] = actual
                        zeroed[dof_id] = True
                        print(f"  DOF {dof_id} zeroed at count: {controller.zero_offset[dof_id]}")
                else:
                    consecutive[dof_id] = 0

    def move_to_halfway(dof_ids):
        sids   = [cfg[d]["servo_id"] for d in dof_ids]
        counts = [controller._mm_to_counts(d, cfg[d]["max_mm"] / 2) for d in dof_ids]
        controller.servo.set_positions(sids, counts, transit_speed, transit_acc, transit_torque)
        start = time.time()
        while time.time() - start < transit_timeout:
            if all(abs(controller.servo.read_position(cfg[d]["servo_id"]) - c) <= transit_tolerance
                   for d, c in zip(dof_ids, counts)
                   if controller.servo.read_position(cfg[d]["servo_id"]) is not None):
                print(f"  DOFs {dof_ids} reached halfway.")
                return
            time.sleep(0.02)
        print(f"  Warning: move_to_halfway timed out for DOFs {dof_ids}")

    key = _config_key(controller)
    print(f"[{key}] Phase 1: closing x-direction...")
    run_phase(DOF_PHASE1)
    print(f"[{key}] Phase 1: moving x to halfway...")
    move_to_halfway(DOF_PHASE1)

    print(f"[{key}] Phase 2: closing y-direction...")
    run_phase(DOF_PHASE2)
    print(f"[{key}] Phase 2: moving y to halfway...")
    move_to_halfway(DOF_PHASE2)

    print(f"[{key}] Phase 3: closing z-direction...")
    run_phase(DOF_PHASE3)
    print(f"[{key}] Phase 3: moving z to halfway...")
    move_to_halfway(DOF_PHASE3)

    controller.is_zeroed = True
    print(f"[{key}] All DOFs zeroed. zero_offset: {controller.zero_offset.tolist()}")

    if save:
        save_offsets(controller, save_path)


def zero_single(controller, dof_id: int, stall_threshold: int = 5,
                confirm_count: int = 5, zero_torque: int = 50, creep_speed: int = 50):
    """Drive a single DOF to hard stop. Debug use — does not set is_zeroed."""
    cfg         = controller.dof_config
    sid         = cfg[dof_id]["servo_id"]
    orientation = cfg[dof_id]["orientation"]

    current_counts = controller.servo.read_position(sid)
    if current_counts is None:
        raise RuntimeError(f"DOF {dof_id} not responding")

    print(f"Single DOF zero — DOF {dof_id}, starting counts: {current_counts}")
    far_target = current_counts - int(orientation * 10000)
    controller.servo.set_position(sid, far_target, speed=creep_speed, acc=20, torque=zero_torque)

    prev_actual = current_counts
    consecutive = 0
    while True:
        time.sleep(0.1)
        actual   = controller.servo.read_position(sid)
        movement = abs(actual - prev_actual) if actual is not None else 999
        print(f"  actual: {actual} | movement: {movement}")
        prev_actual = actual
        if movement < stall_threshold:
            consecutive += 1
            if consecutive >= confirm_count:
                controller.zero_offset[dof_id] = actual
                print(f"DOF {dof_id} zeroed at: {controller.zero_offset[dof_id]}")
                return
        else:
            consecutive = 0


if __name__ == "__main__":
    from cartesian_hand import CartesianHand, PORT_1, PORT_2, config_1, config_2

    parser = argparse.ArgumentParser(description="Zero gripper DOFs to hard stop")
    parser.add_argument("--config", choices=["1", "2"], default="2",
                        help="Which config to use: 1 or 2 (default: 2)")
    parser.add_argument("--load",  action="store_true",
                        help="Load saved offsets instead of physical zeroing")
    parser.add_argument("--dof",   type=int, default=None,
                        help="Single DOF id to zero (omit for full sequence)")
    parser.add_argument("--zero-torque", type=int, default=150)
    parser.add_argument("--creep-speed", type=int, default=50)
    args = parser.parse_args()

    port   = PORT_1   if args.config == "1" else PORT_2
    config = config_1 if args.config == "1" else config_2

    with CartesianHand(port, config) as hand:
        if args.load:
            if not load_offsets(hand):
                print("Load failed — run without --load to zero physically.")
        elif args.dof is not None:
            zero_single(hand, args.dof,
                        zero_torque=args.zero_torque, creep_speed=args.creep_speed)
        else:
            zero_all(hand, zero_torque=args.zero_torque, creep_speed=args.creep_speed)
        hand.release()
