import random


class EnvAdapter:

    def __init__(self, env, render=False):
        self.env = env
        self.render = render

    def reset(self):
        raise NotImplementedError('reset method must be implemented')

    def step(self, action) -> (object, float, bool):
        """must return observation, reward (float) and if done or not (bool)"""
        raise NotImplementedError('step method must be implemented')

    def get_n_actions(self) -> int:
        """must return number of actions of the current env"""
        raise NotImplementedError('get_n_actions method must be implemented')

    def get_random_action(self):
        """must return a random action"""
        raise NotImplementedError('get_random_action method must be implemented')


class GymEnvAdapter(EnvAdapter):

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)
        return observation, float(reward), done, info

    def get_n_actions(self):
        try:
            return self.env.action_space.n
        except AttributeError:
            return self.env.action_space.shape[0]

    def get_random_action(self):
        return self.env.action_space.sample()


class PleEnvAdapter(EnvAdapter):
    """Pygame learning env adapter"""

    def reset(self):
        self.env.reset_game()
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        return observation

    def step(self, action) -> (object, float, bool):
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        reward = self.env.act(self.env.getActionSet()[action])
        done = self.env.game_over()
        return observation, reward, done, {}

    def get_n_actions(self) -> int:
        return len(self.env.getActionSet())

    def get_random_action(self):
        return random.randint(0, len(self.env.getActionSet()) - 1)
