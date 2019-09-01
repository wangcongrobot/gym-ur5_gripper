import os
import imageio
import gym
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG, SAC
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym_ur5_gripper

from gym_ur5_gripper.algos.utils.algo_utils import plot_results


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('UR5Gripper-v0')
env = DummyVecEnv([lambda: env])
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
# model = DDPG(LnMlpPolicy, env,  verbose=1, tensorboard_log=log_dir)
model = SAC('MlpPolicy', env, verbose=1)
model.load(log_dir + 'sacbest_model.pkl')

plot_results('/tmp/gym/sac/')

for i in range(20):
  obs = env.reset()
  env.render('human')
  for i in range(300):
    # action = env.action_space.sample()
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render('human')

# images = []
# obs = model.env.reset()
# img = model.env.render(mode='rgb_array')
# for i in range(500):
#   images.append(img)
#   action, _ = model.predict(obs)
#   obs, _, _, _ = model.env.step(action)
#   img = model.env.render(mode='rgb_array')

#   imageio.mimsave('ur5_gripper_sac.gif', [np.array(img[0]) for i, img in enumerate(images) if i%2 ==0], fps=29)

