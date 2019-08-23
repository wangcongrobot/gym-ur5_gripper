# gym-ur5_gripper

This is a gym environment, using ur5 robot and robotiq 3 finger gripper, in mujoco simulator.

This package includes:

- Robot environments
  - ur5 env
  - gripper env
- Task environments
  - UR5GripperCatchBall-v0
  - UR5GripperPickAndPlace-v0
  - UR5GripperSlide-v0
  - UR5GripperReach-v0
  - UR5GripperPush-v0

## install

```bash

$ git clone https://github.com/wangcongrobot/gym-ur5_gripper.git
$ cd gym-ur5_gripper
$ virtualenv env --python=python3
$ pip install -e .
$ python gym-ur5_gripper/tests/test_ur5_gripper_env.py

```

## Robot Environments

### UR5 Env

- Action space:
  - Cartesian space: hand 6D pose (x,y,z), (w,x,y,z)
  - Joint space: torse and arm joint positions


- Observation space:
  - arm joint positions
  - end-effector pose
  - end-effector velocity



## Task Environments

### UR5 Reach Env

####Observation space

- UR5's observation space
- plus object position and orientation

#### Reward function

In this environment, the reward function is given by:

- the distance between the end-effector and the desired position
- plus a bonus when the end-effector is close to the desired position

Here is the code used to compute the reward function:

```python

reward = np.float(32.0)
objPos, objOrn = p.getBsePositionAndOrientation(self._objID)
endEffAct = self._panda.getObservation()[0:3]
d = goal_distance(np.array(endEffAct), np.array(objPos))
reward = -d
if d <= self._target_dist_min:
    reawrd = np.float32(1000.0) + (100 - d*80)
    return reward

```


