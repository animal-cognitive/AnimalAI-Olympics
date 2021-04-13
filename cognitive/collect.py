import gym
import ray
import ray.rllib.agents.a3c as a3c
from ray import tune
from cognitive.primitive_arena import *


def train_agent():
    ray.init(include_dashboard=False)
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.0001]),  # ([0.01, 0.001, 0.0001]),
    }
    analysis = tune.run(
        "A3C",
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
    return checkpoints


def access_agent():
    ray.init(include_dashboard=False)
    config = {
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
    }
    # checkpoint_path = checkpoints[0][0]
    agent = a3c.A3CTrainer(config=config)
    # agent.restore('D:\\Git\\AnimalAI-Olympics\\log\\PPO\\PPO_CartPole-v0_314b5_00000_0_lr=0.0001_2021-03-19_15-32-15\\checkpoint_000005\\checkpoint-5')
    agent.restore(
        'D:\\Git\\AnimalAI-Olympics\\log\\A3C\\A3C_CartPole-v0_b1b96_00000_0_lr=0.0001_2021-03-19_15-50-10\\checkpoint_000040\\checkpoint-40')
    env = gym.make('CartPole-v0')
    obs = env.reset()
    sum_reward = 0
    for i in range(1000):
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            print(agent.get_policy().model)
            print(agent.get_policy())
            print(agent.get_policy().model.base_model.summary())
            break

class DIRCollection:
    def __init__(self):
        self.modes = [BeforeOrBehind, Occlusion, Rotation]

    def run(self, mode, agent):
        bob = Rotation()
        arena_config, con = bob.generate_config()
def collect_all(ds_size=2):
    checkpoint = str(PROJECT_ROOT / 'log/PPO_unity_env_2021-04-09_02-16-38lwq8v0v4/checkpoint_3500/checkpoint-3500')
    sd = unpack_checkpoint(checkpoint)
    collect(sd, Arena=BeforeOrBehind, ds_size=ds_size)
    collect(sd, Arena=Occlusion, ds_size=ds_size)
    collect(sd, Arena=Rotation, ds_size=ds_size)


def unpack_checkpoint(checkpoint):
    checkpoint = pickle.load(open(checkpoint, "rb"))
    objs = pickle.loads(checkpoint["worker"])
    sd = objs["state"]["default_policy"]
    for k, v in sd.items():
        if isinstance(v, np.ndarray):
            sd[k] = torch.from_numpy(v)
    return sd

if __name__ == '__main__':
    access_agent()
