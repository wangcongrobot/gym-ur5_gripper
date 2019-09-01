import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

import gym_ur5_gripper

import ray
ray.init()

env = gym.make('UR5Gripper-v0')
env.render('human')
env = DummyVecEnv([lambda: env])

model = SAC(MlpPolicy, env, verbose=1)
@ray.remote()
def sac_learn():
    model.learn(total_timesteps=500000, log_interval=100)
sac_learn.remote()
model.save("sac_ur5_gripper")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_ur5_gripper")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()