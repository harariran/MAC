from gym import Wrapper
from gym.spaces import MultiDiscrete
# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np


class SingleTaxiWrapper(Wrapper):
    """
    A wrapper for multi-taxi environments aligning the environments'
    API with OpenAI Gym if using only 1 taxi.
    """

    def __init__(self, env):
        assert env.num_taxis == 1
        super().__init__(env)

    def reset(self):
        # run `reset` as usual.
        # returned value is a dictionary of observations with a single entry
        obs = self.env.reset()

        # a, b, _, c, d = self.unwrapped.state
        # self.unwrapped.state = [a, b, [[0, 0]], c, d]

        # return the single entry value as is.
        # no need for the key (only one agent)
        ret = next(iter(obs.values()))

        return ret

    def step(self, action):
        # step using "joint action" of a single agnet as a dictionary
        step_rets = self.env.step({self.env.taxis_names[0]: action})

        # unpack step return values from their dictionaries
        return tuple(next(iter(ret.values())) for ret in step_rets)


class SinglePassengerPosWrapper(Wrapper):
    '''Same for above, when we also have only one passenger'''

    def __init__(self, env, pass_pos):
        super().__init__(env)
        self.__pass_pos = pass_pos

    def reset(self):
        obs = self.env.reset()
        a, b, _, c, d = self.unwrapped.state
        self.unwrapped.state = [a, b, [[0, 0]], c, d]
        obs[2:4] = self.__pass_pos
        return obs


class TaxiObsPrepWrapper(Wrapper):
    '''Preprocess observations to make data more meaningful for deep networks - encode passenger locations as vectors
    (one hot) and scale according to coordinate system dimensions.'''

    def __init__(self, env):
        super().__init__(env)
        self.map_h = len(self.unwrapped.desc) - 2 - 1
        self.map_w = (len(self.unwrapped.desc[0]) - 1) // 2 - 1

    def reset(self):
        return self._obs_prep(self.env.reset())

    def step(self, action):
        obs, r, d, i = self.env.step(action)
        return self._obs_prep(obs), r, d, i

    def _obs_prep(self, obs):
        taxi_and_pass = obs[:-self.unwrapped.num_passengers].astype(np.float64)
        pass_stat = obs[-self.unwrapped.num_passengers:]

        taxi_and_pass[::2] = taxi_and_pass[::2] / self.map_h
        taxi_and_pass[1::2] = taxi_and_pass[1::2] / self.map_w
        pass_stat = one_hot(torch.from_numpy(pass_stat).to(torch.int64) - 1, num_classes=3).flatten().numpy().astype(
            np.float64)

        return np.concatenate([taxi_and_pass, pass_stat])

    @property
    def observation_space(self):
        obs_space_v = self.env.observation_space.nvec
        taxi_pass_info = obs_space_v[:-self.unwrapped.num_passengers]
        new_obs_space_v = np.concatenate([taxi_pass_info,
                                          [2] * (self.unwrapped.num_taxis + 2) * self.unwrapped.num_passengers])

        return MultiDiscrete(new_obs_space_v)

class EnvWrappper:
    def __init__(self, env, env_agents, num_observation_spaces=1, num_actions=1):
        self.env = env
        self.env_agents = env_agents
        self.num_observation_spaces = num_observation_spaces
        self.num_actions = num_actions

    def get_env(self):
        return self.env

    def get_num_obs(self):
        return self.num_observation_spaces

    def get_num_actions(self):
        return self.num_actions

    def get_env_agents(self):
        return self.env_agents

    def step(self, joint_action):
        return self.env.step(joint_action)


class EnvWrappperGym:

    def __init__(self, env, needs_conv=False):
        super(EnvWrappperGym, self).__init__(env, self.env.possible_agents,
                                             self.env.observation_spaces[env.possible_agents[0]].shape,
                                             self.env.action_spaces[env.possible_agents[0]].n)
        self.needs_conv = needs_conv

    def get_needs_conv(self):
        return self.needs_conv
