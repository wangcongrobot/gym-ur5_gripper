from gym.envs.registration import register


# Mujoco
# ---------------------------------------------------

register(
    id='UR5Gripper-v0',
    entry_point='gym_ur5_gripper.envs:UR5GripperEnv',
    max_episode_steps=200,
)

register(
    id="HDTArmPickAndPlace-v0",
    entry_point='gym_ur5_gripper.envs:HDTArmPickAndPlaceEnv',
    max_episode_steps=200,
)