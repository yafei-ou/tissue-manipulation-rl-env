from collections import deque
import itertools
from typing import Any
from gym import Wrapper, Env, ObservationWrapper
from gym import spaces

import numpy as np


class ObservationSqueezeWrapper(ObservationWrapper):

    def observation(self, observation):
        observation = observation[0]
        return observation

    def step(self, action):
        s, r, d, info = super().step(action)
        return s, r, d, info

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

class StateAugmentContinuousActionWrapper(Wrapper):
    def __init__(self, env: Env, padding_size = 4) -> None:
        super().__init__(env)

        self.action_space = env.action_space
        obs_size = env.observation_space.shape
        action_size = env.action_space.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size[0] + action_size[0]*padding_size, ))
        print(self.observation_space)

        self.past_actions = deque(maxlen=padding_size)
        self.padding_size = padding_size

    def reset(self, *args, **kwargs) -> Any:
        obs = self.env.reset(*args, **kwargs)
        
        for _ in range(self.padding_size):
            act = self.action_space.sample()
            self.past_actions.appendleft(act)

        return self.get_obs_aug(obs)

    def step(self, action):
        _action = action.copy()
        obs, reward, done, info = self.env.step(_action)
        obs_aug = self.get_obs_aug(obs)
        self.past_actions.appendleft(_action)
        return obs_aug, reward, done, info

    def get_obs_aug(self, obs):
        obs = obs.squeeze()
        actions = np.array(list(self.past_actions)).flatten().squeeze()

        obs_augmented = np.concatenate((obs, actions)).flatten()

        return obs_augmented
