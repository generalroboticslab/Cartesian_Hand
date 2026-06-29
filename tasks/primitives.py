import time

def squeeze(controller, dof_ids, squeeze_torque):
    controller.enable()
    with controller.lock:
        for d in dof_ids:
            controller._torque[d] = squeeze_torque
            controller.target[d]  = 0.0


def squeeze_until_stall(controller, dof_ids, squeeze_torque,
                        stall_threshold=0.5, confirm_count=3, timeout=10.0):
    squeeze(controller, dof_ids, squeeze_torque)
    prev        = {d: controller.actual[d] for d in dof_ids}
    consecutive = {d: 0     for d in dof_ids}
    stalled     = {d: False for d in dof_ids}
    start = time.time()
    while not all(stalled.values()):
        if time.time() - start > timeout:
            print(f"squeeze_until_stall: timeout for DOFs {[d for d in dof_ids if not stalled[d]]}")
            break
        time.sleep(0.05)
        with controller.lock:
            current = {d: controller.actual[d] for d in dof_ids}
        for d in dof_ids:
            if stalled[d]:
                continue
            if abs(current[d] - prev[d]) < stall_threshold:
                consecutive[d] += 1
                if consecutive[d] >= confirm_count:
                    stalled[d] = True
                    print(f"  DOF {d} stalled at {current[d]:.1f}mm")
            else:
                consecutive[d] = 0
            prev[d] = current[d]
    print("All DOFs stalled — holding squeeze.")