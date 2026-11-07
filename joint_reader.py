
import csv
import numpy as np
import mujoco

from orca_sim import OrcaHandRight

# Load the ORCA Hand simulation
env = OrcaHandRight(render_mode="human")

try:
    obs, info = env.reset()

    model = env.model
    data = env.data

    # Identify joints with one position and velocity value.
    joint_info = []

    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]

        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue

        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )

        qpos_index = int(model.jnt_qposadr[joint_id])
        qvel_index = int(model.jnt_dofadr[joint_id])

        joint_info.append(
            (name or f"joint_{joint_id}",
             qpos_index, qvel_index)
        )

    print("ORCA HAND JOINT INFORMATION")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"qpos size: {model.nq}")
    print(f"qvel size: {model.nv}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")

    print("\nJOINT NAME / POSITION / VELOCITY")

    for name, pos_idx, vel_idx in joint_info:
        print(
            f"{name:30s} "
            f"Position: {data.qpos[pos_idx]:8.4f} "
            f"Velocity: {data.qvel[vel_idx]:8.4f}"
        )

    # Save multiple readings to a CSV file.
    with open("joint_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step","time_s", "joint_name",
            "position", "velocity"
        ])

        for step in range(100):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            for name, pos_idx, vel_idx in joint_info:
                writer.writerow([
                    step,
                    float(data.time),
                    name,
                    float(data.qpos[pos_idx]),
                    float(data.qvel[vel_idx])
                ])

            if terminated or truncated:
                obs, info = env.reset()

    print("\nJoint data saved to joint_data.csv")

finally:
    env.close()
