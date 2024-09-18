import copy
MODELS_PATH = "C:/Users/Lab-user-3/ran_py_git/copies/MAC_planner_com/models"



import numpy
import torch as th
from multi_taxi import ObservationType
from multi_taxi.env import reward_tables
from torch.distributions import Categorical

from src.agents.agent import Agent, DecisionMaker
from src.environments.env_wrapper import*
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt

from multi_taxi.env import multi_taxi_v0 as TaxiEnv
from multi_taxi.env import single_taxi_v0 as SingleTaxi
from gym import Wrapper
import gymnasium as gym
from gym.spaces import MultiDiscrete, Box, Discrete
from gym.spaces.discrete import Discrete as Dis




# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN, PPO
# from stable_baselines3 import *
# from stable_baselines3.common.policies import obs_as_tensor
# from stable_baselines3.common.distributions import Distribution

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()




# def predict_proba(model, observation) -> Distribution:
#     obs = obs_as_tensor(observation, model.policy.device)
#     dis = model.policy.get_distribution(obs)
#     probs = dis.distribution.probs
#     probs_np = probs.detach().numpy()
#     return probs_np


class Discrete_cast(Dis):
    def __init__(self, disc : Discrete):
        super().__init__(disc.n)

class PPODecisionMaker(DecisionMaker):
    def __init__(self, action_space, env, agent_name = 'taxi_0', train_more = False, total_timesteps = 20000, learning_rate = 0.003):
        self.space = action_space
        self.env = env
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        # self.demo_env = copy.deepcopy(env)
        self.init_state = copy.deepcopy(env.state())
        self.learning_rate = learning_rate
        try:
            i=1
            custom_objects = {'learning_rate': self.learning_rate}
            while True:
                file_name = 'M_Taxi_PPO' + str(i)
                self.model = PPO.load(MODELS_PATH + '/' + file_name)
                # self.model = PPO.load("C:/Users/Lab-user-3/ran_py_git/copies/MAC_planner_com/models/M_Taxi_PPO1")
                self.model_file_name = file_name
                i+=1
        except:
            pass

        if i>1:
            if train_more:
                self.model.set_env(env)
                self.model.learn(total_timesteps=self.total_timesteps,log_interval=2)
                self.model.version = i
                new_model_file_name = 'M_Taxi_PPO' + str(self.model.version)
                self.model.save(MODELS_PATH + '/'  + new_model_file_name)
        else:
            self.train_model()

        # for debug:
        # print(f"{self.model.observation_space}")
        # print(f"model {self.model.obs_meanings}")
        # print(f"env {self.env.unwrapped.get_observation_meanings(agent_name)}")


        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

# fix observation space if not fit to trained dimension
    def fit_obs(self, obs):
        temp = [obj for obj in obs[:(self.obs_size-1)]]
        temp.append(obs[-1])
        print(f"{temp}")
        return temp

    # fix observation space if not fit to trained dimension
    def fit_obs_from_multi_to_single(self, obs):
        new_obs = []
        try:
            new_meanings = self.env.unwrapped.get_observation_meanings()
        except:
            new_meanings = self.env.unwrapped.get_observation_meanings(self.agent_name)

        #check if MIXED obs
        if isinstance(new_meanings,dict):
            try:
                new_meanings = new_meanings['symbolic']
            except:
                pass

        for iter in range(len(new_meanings)):
            if new_meanings[iter] in self.model.obs_meanings:
                new_obs.append(obs[iter])

        return numpy.asarray(new_obs)

    def get_action(self, observation):

        if not isinstance(self.model.observation_space, type(observation)):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            if isinstance(observation,dict):
                if 'symbolic' in observation.keys():
                    observation = observation['symbolic']

        if len(observation) > self.model.observation_space.shape[0]:
            # print(observation)
            observation = self.fit_obs_from_multi_to_single(observation)
            # print(observation)

        # if not self.__is_image_obs:
        #     if (len(observation)!=self.obs_size):
        #         observation = self.fit_obs(observation)
        action, _states = self.model.predict(observation)

        #  precict values of actions
        # first - transfer obs. to tensor
        o, is_vec = self.model.policy.obs_to_tensor(observation)
        # get the prob class out of the model policy
        prob = self.model.policy.get_distribution(o)
        # get the distribution as probability percentage
        probs = prob.distribution.probs
        # get the log values of each action
        a = self.model.policy.evaluate_actions(o,th.Tensor([range(0,5)]))


        return action

    def train_model(self):
        if self.env.unwrapped.num_taxis==1:
            temp_env = self.env
        else:
            temp_env = SingleTaxi.gym_env(
                                   num_passengers=self.env.unwrapped.num_passengers,
                                   pickup_only=self.env.unwrapped.pickup_only,
                                   observation_type=self.env.unwrapped.observation_type,
                                   field_of_view=self.env.unwrapped.field_of_view,
                                   render_mode=self.env.unwrapped.render_mode)

        try:
            self.obs_meanings = temp_env.unwrapped.get_observation_meanings()
        except:
            self.obs_meanings = temp_env.unwrapped.get_observation_meanings('taxi_0')

        temp_env.reset()
        # temp_env.observation_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        # temp_env.action_space = temp_env.action_spaces[temp_env.possible_agents[0]]
        # temp_env.action_space = Discrete(temp_env.action_space.n)
        temp_env.env.render()

        model = PPO("MlpPolicy", temp_env, learning_rate=0.00001,n_steps=1024, verbose=1)
        model.learn(total_timesteps=self.total_timesteps,log_interval=2)
        model.obs_meanings = self.obs_meanings
        self.model = model
        self.model.save(self.model_file_name,MODELS_PATH)


def from_RGBarray_to_image(obs):
    fig = plt.figure(figsize=(16, 4))
    for ob in obs:
        plt.imshow(obs)
        # if filename is None:
        plt.show()
    # ax = fig.add_subplot(1, len(self.agents), i)
    # i += 1
    # plt.title(title)
    # ax.imshow(observation[agent_name])
    return PIL.Image.frombytes('RGB',
                        fig.canvas.get_width_height(), fig.canvas.tostring_rgb())



if __name__ == '__main__':
    # check code:
    env = SingleTaxi.gym_env(
                               num_passengers=3,
                               pickup_only=False,
                               observation_type=ObservationType.SYMBOLIC,
                               field_of_view=[None],
                               render_mode='human')
    # env = TaxiEnv(num_taxis=1, pickup_only=True, observation_type='symbolic')  # pickup_only=True,

    # env = SinglePassengerPosWrapper(env, pass_pos=[0, 0])

    # env = EnvWrappper(env, env.agents)
    #
    # print('EnvironmentWrapper created')

    obs = env.reset()


    # check code for setting state
    #     p = copy.deepcopy(env.unwrapped.state().passengers)
    #     t = copy.deepcopy(env.unwrapped.state().taxis)
    #     all = p + t
    #     obs = env.reset()
    #     s = env.unwrapped.state()
    #     env1 = env_pos_change(env, all)
    #     s1 = env1.unwrapped.state()

    agents = env.env.agents
    agent_name = agents[0]
    m = env.unwrapped.get_action_meanings()

    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = PPODecisionMaker(env.action_space,env, agent_name,train_more=True, total_timesteps=5000)
    env.render()
    print(f"obs:{obs}")
    act = int(D_M.get_action(obs[0]))

    # a = D_M.model.policy.predict_values(obs[0])

    print(f"next action: { m[act] }")



