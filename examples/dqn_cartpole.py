import numpy as np
import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.callbacks import WandbLogger
from rl.model_adapters import GymEnvAdapter, PleEnvAdapter
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from ple.games.flappybird import FlappyBird
from ple.games.catcher import Catcher
from ple import PLE

# env = PLE(Catcher(), display_screen=False, force_fps=True)
# env.init()
#
# env_adapter = PleEnvAdapter(env=env)

env_adapter = GymEnvAdapter(env=gym.make('LunarLander-v2'))

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1, 8)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(env_adapter.get_n_actions()))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(
    model=model,
    nb_actions=env_adapter.get_n_actions(),
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    policy=policy
)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env_adapter, nb_steps=200_000, visualize=True, verbose=2, callbacks=[WandbLogger()])

dqn.save_weights('dqn_flappy_weights.h5f', overwrite=True)

dqn.test(env_adapter, nb_episodes=5, visualize=True)
