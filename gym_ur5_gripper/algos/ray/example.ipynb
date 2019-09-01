import gym
import numpy as np
import time

# serial version
def run1():
    env = gym.make("CartPole-v0")
    env.reset()
    steps = []
    for _ in range(1000):
        steps += [env.step(env.action_space.sample())]
    return len(steps)

%time result = [run1() for i in range(100)]
print(sum(result))


# parallel version
import ray
ray.init()

@ray.remote
def run2():  # same as run1 
    env = gym.make("CartPole-v0")
    env.reset()
    steps = []
    for _ in range(1000):
        steps += [env.step(env.action_space.sample())]
    return len(steps)

# note: maybe run this twice to warmup the system
%time result = ray.get([run2.remote() for i in range(100)])
print(sum(result))
