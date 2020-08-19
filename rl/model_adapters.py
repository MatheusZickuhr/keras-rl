import os
import random

import gym
from ple import PLE
from ple.games import Pong, Catcher, Pixelcopter, FlappyBird, MonsterKong, PuckWorld, RaycastMaze, Snake, WaterWorld


class EnvAdapter:

    def __init__(self, env_name, render=False):
        self.env_name = env_name
        self.render = render
        self.env = None

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

    def get_input_shape(self):
        raise NotImplementedError('get_input_shape method must be implemented')


class GymEnvAdapter(EnvAdapter):

    def __init__(self, *args, **kwargs):
        super(GymEnvAdapter, self).__init__(*args, **kwargs)
        self.env = gym.make(self.env_name)

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

    def get_input_shape(self):
        return self.env.observation_space.shape


envs_lookup_table = {
    'pong': Pong,
    'catcher': Catcher,
    'pixelcopter': Pixelcopter,
    'flappybird': FlappyBird,
    'monsterkong': MonsterKong,
    'puckworld': PuckWorld,
    'raycastmaze': RaycastMaze,
    'snake': Snake,
    'waterworld': WaterWorld,
}


class PleEnvAdapter(EnvAdapter):

    def __init__(self, *args, **kwargs):
        super(PleEnvAdapter, self).__init__(*args, **kwargs)

        if not self.render:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        Game = envs_lookup_table[self.env_name]
        self.env = PLE(Game(), display_screen=self.render, force_fps=not self.render)
        self.env.init()

    def reset(self):
        self.env.reset_game()
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        return observation

    def step(self, action) -> (object, float, bool):
        reward = self.env.act(self.env.getActionSet()[action])
        observation = self.env.getGameState()
        observation = [val for key, val in observation.items()]
        done = self.env.game_over()
        return observation, reward, done, {}

    def get_n_actions(self) -> int:
        return len(self.env.getActionSet())

    def get_random_action(self):
        return random.randint(0, len(self.env.getActionSet()) - 1)

    def get_input_shape(self):
        return (len(self.env.getGameState()),)
