#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.bench import Monitor
import stable_baselines.common.tf_util as tf_util

import numpy as np
import os

import gym_ur5_gripper

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 500 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(os.path.join(output_dir,'log')), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
            if (n_steps + 1) % 1000 == 0:
                # Save model
                print("Saving model at iter {}".format(x[-1]))
                _locals['self'].save(os.path.join(output_dir, str(x[-1])+'model.pkl'))
    n_steps += 1
    # Returning False will stop training early
    return True

def train(env_id, num_timesteps, seed, model_path=None):
    """
    Train TRPO model for the mujoco environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    global output_dir 
    output_dir = model_path
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure(folder=model_path)
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DEBUG)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        env = make_mujoco_env(env_id, workerseed)
        env = Monitor(env, model_path)
        model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=model_path)
        # or load a model and continue to train
        # model = TRPO("./trpo.pkl", env=env, tensorboard_log=model_path)
        model.learn(total_timesteps=num_timesteps, callback=callback)
        env.close()
        if model_path:
            tf_util.save_state(model_path)

    return model


def main():
    """
    Runs the test
    """
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return:  (ArgumentParser) parser {'--env': 'Reacher-v2', '--seed': 0, '--num-timesteps': int(1e6), '--play': False}

    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    return parse
    """
    env_id = 'UR5Gripper-v0'
    model_path = '/tmp/gym/trpo_mpi/'
    args = mujoco_arg_parser().parse_args()
    # train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    train(env_id=env_id, num_timesteps=int(1e7), seed=0, model_path=model_path)


if __name__ == '__main__':
    main()
