import gym
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


def access_agent():
    import ray
    from ray import tune

    ray.init(include_dashboard=False)
    config = {
            "env": "CartPole-v0",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": tune.grid_search([0.0001]), #([0.01, 0.001, 0.0001]),
        }
    analysis=tune.run(
        "PPO",
        stop={"episode_reward_mean": 120},
        config=config,
        local_dir="log",
        checkpoint_at_end=True
    )

    # list of lists: one list per checkpoint; each checkpoint list contains
    # 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean")
    print(checkpoints)
    env = gym.make('CartPole-v0')
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
    }
    checkpoint_path= checkpoints[0][0]
    agent = ppo.PPOTrainer(config=config)
    agent.restore(checkpoint_path)

if __name__ == '__main__':
    access_agent()