import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec

import gym_ur5_gripper

# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('UR5Gripper-v0')

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env,  verbose=1, tensorboard_log=log_dir)

model.load(log_dir + 'best_model.pkl')

for i in range(20):
  obs = env.reset()
  env.render('human')
  for i in range(300):
    # action = env.action_space.sample()
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render('human')
