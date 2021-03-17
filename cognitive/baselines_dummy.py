from stable_baselines3 import PPO
import animalai
import gym
from animalai.envs.gym.environment import AnimalAIGym
from animalai.envs.arena_config import ArenaConfig


# this is a dummy agent that I can use to interact with the environment.
# the goal is to create a environment and throw it into baselines unity integration.

def main():
    env = AnimalAIGym(
            environment_filename='examples/env/AnimalAI',
            arenas_configurations=ArenaConfig('examples/configurations/curriculum/0.yml')
        )
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # inspect the model, we have access to the states.
        # this is good.
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == '__main__':
    main()