import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

import gym_ur5_gripper

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
trainer = ppo.PPOTrainer(config=config, env="UR5Gripper-v0")

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at: ", checkpoint)