
import os
import numpy as np
import time
import random
from gym import utils, spaces
from gym_ur5_gripper.envs import mujoco_env
from gym_ur5_gripper.envs.utils import mujoco_utils

class UR5GripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, 
                 reward_type=True,
                 n_actions=8,
                 has_object=True):

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'ur5_gripper/ur5_gripper.xml', 5)

        # override the mujoco_env action_space and observation_space
        # self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        
        self.viewer = self._get_viewer('human')

        self.reward_type = reward_type
        # 3(pos of end-effector) + 4(quat of end-effector) + 1(gripper)
        self.n_actions = n_actions
        # (x,y,z)
        self.arm_dof = 3
        self.gripper_dof =1
        self.dof = self.arm_dof + self.gripper_dof

        self.has_object = True

        self.initial_gripper_xpos = None
        self.obj_range = 0.15


    def step(self, action):

        self._set_action(action)
        self.sim.step()
        self.sim.forward()

        obs = self._get_obs()

        done = False
        reward, done = self.compute_reward(action)
        info = {
            'is_success': self._check_success(),
        }

        return obs, reward, done, info

    def viewer_setup(self):
        body_id = self.sim.model.body_name2id('gripperpalm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def compute_reward(self, action):
        # compute sparse rewards
        # self._check_success()
        # reward 

        # add in shaped rewards
        # if self.reward_shaping:
            # staged_rewards = self.staged_rewards()
            # reward += max(staged_rewards)
        staged_rewards = self.staged_rewards(action)
        reward = np.sum(staged_rewards)
        print("one step reward: ", reward)

        done = False
        # if self.sim.data.get_site_xpos('object')[2] < 0.1:
            # done = True
        return reward, done
    
    def reward(self, action):
        staged_rewards = self.staged_rewards(action)
        return np.sum(staged_rewards)

    def staged_rewards(self, action):
        """
        Returns staged rewawrds based on current physical states.
        Stages consist of following, reaching, grasping, lifting, and hovering.
        """
        # importance degree of four shaping reward
        control_mult = -0.01
        reach_mult = 0.1
        grasp_mult = 1.0
        lift_mult = 0.5
        hover_mult = 0.7

        # control reward
        reward_ctrl = np.square(action).sum() * control_mult
        #reward_ctrl = 0.
        print("reward_ctrl: ", reward_ctrl)

        # following reward

        # reaching reward
        obj_pos = self.sim.data.get_site_xpos('object')
        palm_pos = self.sim.data.get_mocap_pos('robot0:mocap')
        target_pos = self.sim.data.get_site_xpos('target')
        dist = np.linalg.norm(palm_pos - obj_pos)
        # the larger distance, the fewer reward
        # convert the dist to range (0, 1)
        reward_reach = (1 - np.tanh(1.0 * dist)) * reach_mult
        print("reward_reach: ", reward_reach)

        # grasping reward
        # int(False) = 0 int(True) = 1
        reward_grasp = int(self._check_grasp()) * grasp_mult
        print("reward_grasp: ", reward_grasp)

        # lifting reward


        return reward_ctrl, reward_reach, reward_grasp


    def _check_grasp(self):
        """
        Return True if the gripper has grasped the object.
        Using the contact detection between gripper and object.
        First, palm contact must be true.
        Second, three (or two) fingers contact with the object can return True.
        """
        # get the contact geom information from ur5_keep_up.xml file
        finger_1_geom_names = ["f1_l0", "f1_l1", "f1_l2", "f1_l3"]
        finger_2_geom_names = ["f2_l0", "f2_l1", "f2_l2", "f2_l3"]
        finger_3_geom_names = ["f3_l0", "f3_l1", "f3_l2", "f3_l3"]
        palm_geom_name = "gripperpalm"
        object_geom_name = "object"

        # get the geom ids from the names
        finger_1_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_1_geom_names
        ]
        finger_2_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_2_geom_names
        ]
        finger_3_geom_ids = [
            self.sim.model.geom_name2id(x) for x in finger_3_geom_names
        ]
        palm_geom_id = self.sim.model.geom_name2id(palm_geom_name)
        object_geom_id = self.sim.model.geom_name2id(object_geom_name)

        # touch/contact detection flag
        touch_finger_1 = False
        touch_finger_2 = False
        touch_finger_3 = False
        touch_palm = False
        has_grasp = False

        # int ncon: number of detected contacts
        for i in range(self.sim.data.ncon):
            # list of all detected contacts
            contact = self.sim.data.contact[i]
            # if the object is in the detected contact geom1
            if contact.geom1 == object_geom_id:
                # wether the finger 1 in the detected contacts
                if contact.geom2 in finger_1_geom_ids:
                    # result: finger 1 touched the object
                    touch_finger_1 = True
                if contact.geom2 in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
                if contact.geom2 in finger_3_geom_ids:
                    # finger 3 touched the object
                    touch_finger_3 = True
                if contact.geom2 == palm_geom_id:
                    # palm touched the object
                    touch_palm = True
            # if the object is in the detected contact geom2
            elif contact.geom2 == object_geom_id:
                if contact.geom1 in finger_1_geom_ids:
                    # finger 1 touched the object
                    touch_finger_1 = True
                if contact.geom1 in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
                if contact.geom1 in finger_3_geom_ids:
                    # finger 3 touched the object
                    touch_finger_3 = True
                if contact.geom1 == palm_geom_id:
                    # palm touched the object
                    touch_palm = True

        # TODO: two finger may also has grasp
        # the palm touch must be true first
        if touch_palm:
            # has three finger touch must be true
            if touch_finger_1 and touch_finger_2 and touch_finger_3:
                has_grasp = True
            # has two finger touch maybe true
            elif touch_finger_1 and touch_finger_3:
                has_grasp = True
            # has two finger touch maybe true
            elif touch_finger_2 and touch_finger_3:
                has_grasp = True
        if has_grasp:
            print("Get a successful grasp!")

        return has_grasp


    def _check_success(self):
        """
        Return True if task has been completed.
        """
        return self._check_grasp()


    def _get_obs(self):
        
        robot_qpos = self.data.qpos.ravel()
        robot_qvel = self.data.qvel.ravel()

        mocap_pos = self.sim.data.get_mocap_pos('robot0:mocap')
        
        object_pos = self.sim.data.get_site_xpos('object')
        # Positional velocity of the object with respect to the world frame
        object_velp = self.sim.data.get_site_xvelp('object') * self.dt
        # Rotational velocity of the object with respect to the world frame
        object_velr = self.sim.data.get_site_xvelr('object') * self.dt

        palm_pos = self.sim.data.get_site_xpos('gripperpalm')
        # Positional velocity of palm
        palm_velp = self.sim.data.get_site_xvelp('gripperpalm') * self.dt

        target_pos = self.sim.data.get_site_xpos('target')

        ## TODO: when should to use the ravel()?
        return np.concatenate([
            robot_qpos,
            robot_qvel,
            mocap_pos,
            object_pos,
            object_velp,
            object_velr,
            palm_pos,
            palm_velp,
            target_pos,
        ])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        
        mujoco_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end-effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431]) + self.sim.data.get_site_xpos('gripperpalm')
        gripper_rotation = np.array([0., 0., 1., 0.]) # fixed oritation to grasp
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripperpalm').copy()

        self._reset_sim()

        return self._get_obs()

    def _reset_sim(self):
        # self.sim.set_state(self.initial_state)
        # self._restart_target()

        # Randomize start position of object.
        # object joint type: free
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object:joint', object_qpos)

        self.sim.forward()
        return True

    # control
    def _set_action(self, action):
        # arm
        self._pre_action_cartesion(action)
        # gripper: close by signal
        dist = np.linalg.norm(self.sim.data.get_mocap_pos('robot0:mocap') - self.sim.data.get_site_xpos('object'))
        if dist < 0.05:
            self.gripper_format_action(-1) # close the gripper

    # arm end-effector cartesion control
    def _pre_action_cartesion(self, action):
        print("_set_action:", action)
        # assert action.shape == (self.n_actions,) # 8
        assert action.shape == (8,)
        self.action = action
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[7:]

        # arm end-effector cartesion control
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)

        arm_action = np.concatenate([pos_ctrl, rot_ctrl])
        print("arm_action: ", action)

        # Apply arm action to simulation.
        # use mocap to control the arm end-effector
        # mujoco_utils.ctrl_set_action(self.sim, action)
        mujoco_utils.mocap_set_action(self.sim, action)

        # gripper joint control
        # rescale normalized action to control ranges
        print("gripper_ctrl: ", gripper_ctrl)
        gripper_action_actual = self.gripper_format_action(gripper_ctrl)
        print("gripper_cation_actual: ", gripper_action_actual)
        ctrl_range = self.sim.model.actuator_ctrlrange
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_action = bias + weight * gripper_action_actual
        print("applied_action: ", applied_action)
        # self.sim.data.ctrl[:] = applied_action[:] # don't use a random action for gripper

    # arm joint control
    def _pre_action_joint(self, action):
        """

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof dimensions should be actuation controls for the gripper.
        """

        # clip actions into valid range
        # 6 arm joint dof plus 1 gripper joint dof
        assert len(action) == self.dof, "environment got invalid action dimension"
        low, high = self.action_spec
        action = np.clip(action, low, high)

        # another control mode: end-effector pose
        arm_action = action[: self.arm_dof]
        # 1 dof to control the grippper: 1 => open, -1 => closed
        gripper_action_in = action[self.arm_dof : self.arm_dof + self.gripper_dof]
        # 11 dof of the actual gripper
        gripper_action_actual = self.gripper_format_action(gripper_action_in)
        # 17 = 6 + 11
        action = np.concatenate([arm_action, gripper_action_actual])
        print("action: ", action)

        # rescale normalized action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_action = bias + weight * action
        print("applied_action: ", applied_action)
        self.sim.data.ctrl[:] = applied_action

    def _post_action(self, action):

        pass

    def grasp_action(self, dist):
        if dist <= 0.05:
            self.gripper_format_action(-1) # close the gripper

    # self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        print(low)
        print(high)
        return low, high

    def gripper_format_action(self, action):
        """ Given (-1,1) abstract control as np-array return the (-1,1) control signals
        for underlying actuators as 1-d np array
        Args:
            action: 1 => open, -1 => closed
        """
        movement = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
        return -1 * movement * action

