# gym_ur5_gripper_env

## basic env

We use cartesion mode to control the ur5, 1-dof mode to control the gripper. So for the action control, it will be devided into two parts: arm control and gripper.

The gripper control first use the man-made control signal, then we will train a model to learn how to grasp the flying ball.

The important parts:

- action
  - arm action: cartesion, random
  - gripper action: 1-dof, trided
- observation
  - robot observation: qpos, qvel
  - object observation

## ur5 arm

control mode: 

- joint mode

6 dof joint value

- cartesion mode

use mocap to control the end-effector
position: (x,y,z)
rotation: (w,x,y,z)

## robotiq 3 finger gripper

control mode:

- joint mode

11 joint

- 1-dof mode

open/close
