# gym-ur5_gripper

This is a gym environment, using ur5 robot and robotiq 3 finger gripper, in mujoco simulator.

The robot xml file is from http://www.mujoco.org/forum/index.php?resources/universal-robots-ur5-robotiq-s-model-3-finger-gripper.22/

This package includes:

- Robot environments
  - ur5 env
  - gripper env
- Task environments
  - UR5GripperCatchBall-v0

## Overview

- [gym-ur5_gripper](#gym-ur5gripper)
  - [Overview](#overview)
  - [install](#install)
  - [Robot Environments](#robot-environments)
    - [UR5 Env](#ur5-env)
  - [Task Environments](#task-environments)
    - [UR5 Reach Env](#ur5-reach-env)
      - [Observation space](#observation-space)
      - [Reward function](#reward-function)
    - [Gripper](#gripper)

---



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
  - Cartesian space: hand 3D pose (x,y,z)
  - Gripper action: open/close


- Observation space:
  - arm joint positions
  - end-effector pose
  - end-effector velocity



## Task Environments

### UR5 Reach Env

#### Observation space

- UR5's observation space
- plus object position and orientation

#### Reward function

In this environment, the reward function is given by:

- the distance between the end-effector and the desired position
- plus a bonus when the end-effector is close to the desired position

Here is the code used to compute the reward function:


### Gripper

The Robotiq 3 finger gripper has 11 dof, the control mode includes torque control, position control and so on.

We use position control, and change the 11-dof joint control into a 1-dof open/close action.

```python

def gripper_format_action(self, action):
    """ Given (-1,1) abstract control as np-array return the (-1,1) control signals
    for underlying actuators as 1-d np array
    Args:
        action: 1 => open, -1 => closed
    """
    movement = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    return -1 * movement * action
```

P.S.: 1 => open, 0 => close, (0,1) => grasp
