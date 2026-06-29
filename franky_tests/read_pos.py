from franky import Robot

robot = Robot("172.16.0.2")

joints = robot.current_joint_state.position
print("[" + ",  ".join(f"{j:.3f}" for j in joints) + "]")