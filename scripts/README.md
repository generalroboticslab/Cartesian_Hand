# Bench scripts

Unsupported one-off tools. They are not part of the control stack, nothing in
`cartesian_hand/` imports them, and their dependencies are deliberately left out
of `pyproject.toml`. Expect to edit them before they run.

| | |
|---|---|
| `bench_servo_check.py` | Drive one servo 200 counts from wherever it is parked. `PORT` and `SERVO_ID` are constants at the top; edit them first. |
| `franka_arm_testing/` | Hand-on-arm experiments. Needs a Franka Panda and the `franky` package. `ee_prompt.py` is an interactive jog prompt, `sequence.py` a recorded pick-and-place. |
| `sync_sim_asset.py` | Re-copies the MuJoCo model into `assets/` from a `legged_env_v2` checkout, which is not public. The generated model is already bundled, so you do not need this. |
