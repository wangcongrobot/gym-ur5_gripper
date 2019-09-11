import os
from gym import utils
from gym_ur5_gripper.envs import hdt_arm_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('hdt_arm', 'pick_and_place.xml')


class HDTArmPickAndPlaceEnv(hdt_arm_env.HDTArmEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        # initial_qpos = {
        #     'robot0:slide0': 0.405,
        #     'robot0:slide1': 0.48,
        #     'robot0:slide2': 1.0,
        #     'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        # }
        hdt_arm_env.HDTArmEnv.__init__(self)
            # self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            # gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            # obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            # initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
