import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import logger
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG, PPO2, TRPO

# SAC
# from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from stable_baselines.ddpg import AdaptiveParamNoiseSpec
# for multiprocessing
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
# from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
# from stable_baselines.common.misc_util import mpi_rank_or_zero


import gym_ur5_gripper

best_mean_reward, n_steps = -np.inf, 0

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID.
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

def make_mujoco_env(env_id, seed, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    set_global_seeds(seed + 1)
    env = gym.make(env_id)
    # env = Monitor(env, os.path.join(logger.get_dir(), str(rank)), allow_early_resets=allow_early_resets)
    # env = Monitor(env, os.path.join(logger.get_dir(), str(rank)), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True

def evaluate(model, num_steps=1000):
    """
    Helper function to evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(actions)

        # States
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)
        
        mean_rewards = [0.0 for _ in range(env.num_envs)]
        n_episodes = 0
        for i in range(env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])
            n_episodes += len(episode_rewards[i])

        # Compute mean reward
        mean_reward = round(np.mean(mean_rewards), 1)
        print("Mean reward:", mean_reward, "Num episode:", n_episodes)

        return mean_reward


# Create log dir
log_dir = "/tmp/gym/trpo"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env_id = 'UR5Gripper-v0'
num_cpu = 4 # Number of processes to use

env = gym.make('UR5Gripper-v0')
# Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
env = Monitor(env, log_dir, allow_early_resets=True)
# env = SubprocVecEnv([make_mujoco_env(env_id, i) for i in range(num_cpu)])
# env = SubprocVecEnv([lambda: env])
env = DummyVecEnv([lambda: env])

# env = SubprocVecEnv([lambda: gym.make('UR5Gripper-v0') for i in range(num_cpu)])

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
# model = DDPG(MlpPolicy, env, param_noise=param_noise, verbose=1, tensorboard_log=log_dir)
# model = PPO2(MlpPolicy, env, verbose=1)
# model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
# Random Agent, before training
mean_reward_before_train = evaluate(model, num_steps=1000)

# Train the agent
model.learn(total_timesteps=int(1e7), callback=callback)

mean_reward_after_train = evaluate(model, num_steps=1000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()