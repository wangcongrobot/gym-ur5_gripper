import gym
import numpy as np
import matplotlib.pyplot as plt

# import the new env
import gym_ur5_gripper

mode = 'human'
#mode = 'rgb_array'

env = gym.make("FetchPickAndPlace-v1")
# env = gym.make("FetchReach-v1")
print("action space high: ", env.action_space.high)
print("action space low: ", env.action_space.low)
num_actuator = env.sim.model.nu
print('num_actuator: ', num_actuator)
env.render('human')
#env = gym.wrappers.Monitor(env, './video', force=True)
# plt.imshow(env.render(mode='rgb_array', camera_id=-1))
#plt.show()


# plt.show()

for i in range(20):
  env.reset()
  env.render('human')
  for i in range(200):
    action = env.action_space.sample()
    action = np.array([0., 0.4, 0.4, 0.8])
    # print("action_space:", env.action_space)
    # print("action space sample:", action)
    obs, reward, done, info = env.step(action)
    # print("observation:", obs)
    # print("reward:", reward)
    # print("done:", done)
    # print("info:", info)
    env.render('human')
    # print("number actuator: ", num_actuator)
    # print("name: ", env.sim.model.name_actuatoradr)
    # print("actuator contrl range: ", env.sim.model.actuator_ctrlrange
    if done:
          break
