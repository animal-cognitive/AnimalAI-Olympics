import pickle

import gym
import ray
import ray.rllib.agents.a3c as a3c
from ray import tune

from cachey.reduced_cache_model import MyCNNRNNModel
from cachey.load_trained_model import load_trainer
from cognitive.primitive_arena import *

import ray
import ray.rllib.agents.a3c as a3c

from cachey.config import get_cfg
from cognitive.primitive_arena import Occlusion
from ray.rllib.models import ModelCatalog
from cachey.custom_model import *
import subprocess
import time


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


# class DIRCollection:
#     def __init__(self):
#         self.modes = [BeforeOrBehind, Occlusion, Rotation]
#
#     def run(self, mode, agent):
#         bob = Rotation()
#         arena_config, con = bob.generate_config()

def register_models():
    # ModelCatalog.register_custom_model("my_fc_model", MyCNNRNNModel)
    ModelCatalog.register_custom_model("my_rnn_model", MyRNNModel)
    ModelCatalog.register_custom_model("my_convgru_model", MyConvGRUModel)  # NOTE: Only works with image observations.
    ModelCatalog.register_custom_model("mm", MyCNNRNNModel)  # NOTE: Only works with image observations.


def collect(Arena: ArenaManager = BeforeOrBehind, ds_size=1000):
    register_models()
    # ray.init(include_dashboard=False)
    env_config = ppo.DEFAULT_CONFIG.copy()
    arena = Arena()
    arena_config, settings = arena.generate_config()
    env_config_ = {
        "env": GymFactory(arena_config),
        "num_gpus": 1,
        "num_workers": 0,
        "framework": 'torch',
        "model": {
            "custom_model": 'mm',
        },
        "log_level": 'INFO',
    }
    env_config.update(env_config_)
    trainer = ppo.PPOTrainer(config=env_config)
    # trainer.restore()

    input, target = [], []
    for i in range(ds_size):
        a, b = arena.collect_dir(trainer, arena_config, settings, env_config)
        input += a
        target += b
    dataset = DIRDataset(input, target)
    trainer.cleanup()
    # ray.shutdown()
    return dataset


def collect_trainer(trainer, Arena: ArenaManager = BeforeOrBehind, num_envs=10, ds_size_per_env=1000):
    arena = Arena()
    arena_config, settings = arena.generate_config()
    input, target = [], []
    name = {BeforeOrBehind: "BeforeOrBehind",
            Occlusion: "Occlusion",
            Rotation: "Rotation"}
    # None
    # env = trainer.workers.local_worker().env
    for i in range(num_envs):
        a, b = arena.collect_dir(trainer, ds_size_per_env, arena_config, settings, trainer.config)
        input += a
        target += b
    dataset = DIRDataset(input, target)
    with open('cognitive/dataset/' + name[Arena] + ".dir", "wb") as f:
        pickle.dump(dataset, f)
    return dataset


@ray.remote
def par_helper(model, Arena, reuse=10):
    arena = Arena()
    trainer = load_trainer(model)

    arena_config, settings = arena.generate_config()
    while True:
        try:
            x, y = arena.collect_dir(trainer, reuse, arena_config, settings, trainer.config)
        except UnityCommunicationException:
            pass
        except ArenaConfigRegenerationRequest:
            print("Arena config regenerated")
            arena_config, settings = arena.generate_config()
        else:
            break
    trainer.cleanup()
    return x, y


def par(model, num_envs=500, ds_size_per_env=10):
    print('Total dataset size = ' + str(num_envs * ds_size_per_env))
    # Arenas = {BeforeOrBehind: "BeforeOrBehind",
    #           Occlusion: "Occlusion",
    #           Rotation: "Rotation"}
    Arenas = {BeforeOrBehind: "BeforeOrBehind",
              Occlusion: "Occlusion",
              # Rotation: "Rotation"
              }
    # None
    # env = trainer.workers.local_worker().env
    restart_per_env = 50
    restart = num_envs // restart_per_env

    for Arena, name in Arenas.items():
        xs = []
        ys = []
        for _ in tqdm(range(restart)):
            ray.init(num_cpus=6, log_to_driver=False, include_dashboard=False)
            results = [par_helper.remote(model, Arena, ds_size_per_env) for _ in range(restart_per_env)]
            results = [ray.get(r) for r in results]
            for res in results:
                xs += res[0]
                ys += res[1]
            ray.shutdown()
            subprocess.Popen("ray stop --force", shell=True)
            subprocess.Popen("pkill -9 -f AnimalAI.x86_64", shell=True)
            time.sleep(1)
        dataset = DIRDataset(xs, ys)
        dir_path = Path(f"cognitive/{model}_dataset")
        dir_path.mkdir(exist_ok=True)
        with open(f'cognitive/{model}_dataset/{name}.dir', "wb") as f:
            pickle.dump(dataset, f)


#
# def collect_all(ds_size=2):
#     collect(Arena=BeforeOrBehind, ds_size=ds_size)
#     collect(Arena=Occlusion, ds_size=ds_size)
#     collect(Arena=Rotation, ds_size=ds_size)
#

def debug():
    Arenas = {#BeforeOrBehind: "BeforeOrBehind",
              #Occlusion: "Occlusion",
              Rotation: "Rotation"
              }
    ray.init(local_mode=True)
    # mnames = ["cnn", "lstm", "reduced", "whole"]
    mnames = ["reduced"]  # , "whole"]
    for mname in mnames:
        for Arena in Arenas:
            arena = Arena()
            trainer = load_trainer(mname)
            arena_config, settings = arena.generate_config()
            while True:
                try:
                    x, y = arena.collect_dir(trainer, 2, arena_config, settings, trainer.config)
                except UnityCommunicationException:
                    pass
                except ArenaConfigRegenerationRequest:
                    print("Arena config regenerated")
                    arena_config, settings = arena.generate_config()
                else:
                    break
            pass
    ray.shutdown()


def collect_all():
    # 2 hours each collection

    par("cnn")
    par("lstm")
    par("whole")
    par("reduced")


if __name__ == '__main__':
    debug()
