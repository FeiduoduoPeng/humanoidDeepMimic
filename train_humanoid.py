from env_humanoid import DMEnv
import os

import ray
from ray import tune, air
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPOConfig, PPO

def trainRay(model_file=None):
    ray.init()

    config = PPOConfig()
    config = config.resources(num_gpus=1, num_cpus_per_worker=1)
    config = config.framework("torch")

    myModel = {
        "fcnet_hiddens": [1024, 512],
        "fcnet_activation": "relu",
        "use_lstm": False,
    }

    config = config.training(lr=5e-5, gamma=0.99, lambda_=0.95, vf_loss_coeff=2,
        train_batch_size=2048, sgd_minibatch_size=128, num_sgd_iter=10, vf_clip_param=50,
        model=myModel)
    config = config.environment(env=DMEnv, env_config={"randStart": True, "render": False})

    if model_file:
        config = config.rollouts(num_rollout_workers=1, num_envs_per_worker=1)
        config = config.exploration(explore=False)
        algo = config.build()
        algo.restore(model_file)

        env = DMEnv({"randStart": True, "render": True})
        obs = env.reset()

        index = 0
        while True:
            index += 1
            action = algo.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done or index>300:
                index, obs = 0, env.reset()
    else:
        config = config.rollouts(num_rollout_workers=os.cpu_count()-1, num_envs_per_worker=4,
            rollout_fragment_length="auto")
        config = config.to_dict()
        # config["lr"] = tune.grid_search([5e-5, 1e-5, 5e-6])
        tune.run("PPO", name="humanoid", config=config, verbose=1, checkpoint_freq=100, checkpoint_at_end=True,
            stop={"episode_reward_mean": 400, "episode_len_mean": 400}
        )

###############################################################################
###############################################################################
###############################################################################
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model_file")
    args = parser.parse_args()

    if args.model:
        trainRay(args.model)
    else:
        trainRay()