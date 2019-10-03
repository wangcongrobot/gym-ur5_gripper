import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray import tune

ray.init()

tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "FetchPickAndPlace-v1",
        "num_gpus": 3,
        "num_workers": 20,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "eager": False,
    }
)

# config = ppo.DEFAULT_CONFIG.copy()
# config["num_gpus"] = 0
# config["num_workers"] = 20
# config["eager"] = True
# trainer = ppo.PPOTrainer(config=config, env="FetchPickAndPlace-v1")

# # Can optionally call trainer.restore(path) to load a checkpoint.

# for i in range(1000):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))

#    if i % 100 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)