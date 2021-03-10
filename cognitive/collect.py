class CollectData:
    def __init__(self, agent):
        self.agent = None

    def collect(self, arena):
        return FeatureData


class FeatureData:
    def __init__(self):
        pass

    def pickle(self):
        """
        Pickle the feature data
        Also in a separate json store all feature data's environment.
        """
        pass

class Arena:
    def __init__(self):
        pass

def run():
    from animalai.envs.arena_config import ArenaConfig
    from animalai.envs.environment import AnimalAIEnvironment
    from mlagents_envs.exception import UnityCommunicationException
    try:
        environment = AnimalAIEnvironment(
                file_name='examples/env/AnimalAI',
                base_port=5005,
                arenas_configurations=ArenaConfig('examples/configurations/curriculum/0.yml'),
                play=False,
            )
    except UnityCommunicationException:
        # you'll end up here if you close the environment window directly
        # always try to close it from script
        environment.close()

    # if environment:
    #     environment.close()  # takes a few seconds

    import numpy as np

    actions = [[0, 0]] * 50  # Do nothing until the lights come back on
    actions += [[1, 0]] * 40  # Go forward
    actions += [[0, 2]] * 15  # turn left
    actions += [[1, 0]] * 50  # go forward again

    agent_groups = environment.get_agent_groups()
    agent_group_spec = environment.get_agent_group_spec(agent_groups[0])

    agent_group = agent_groups[0]
    visual_observations = []
    velocity_observations = []
    rewards = []
    done = False
    step = 0

    while not done and step < len(actions):
        action = np.array(actions[step]).reshape(1, 2)

        environment.set_actions(agent_group=agent_group, action=action)
        environment.step()
        step_result = environment.get_step_result(agent_group)

        visual_observations.append(step_result.obs[0])
        velocity_observations.append(step_result.obs[1])
        done = step_result.done[0]
        rewards.append(step_result.reward[0])
        max_step_reached = step_result.max_step[0]
        step += 1

    environment.close()

if __name__ == '__main__':
    run()