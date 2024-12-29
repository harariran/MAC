#!/usr/bin/env python
# coding: utf-8

# **Multi-Agent Contorl**

# The purpose of this library is to support different forms of multi-agent control.
# The (current) focus is on setting with a single thread :  a single thread collects the actions to be performed by each agent (the joint actions) and performs it. 
# 
# The difference between the different approaches to control is in the agent that decides which action each agent performs. As can be seen in the image below, the two extreme cases are fully-centralized settings, in which a single agent (i.e. the controller) decide which agent each agent performs (think: dictatorship), and fully-decentralized settings, in which each agent makes its own decision on how to behave (think: anarchy). 
# 

# ![control-specturm.png](data:image/png;base64,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)

# The framework is comprised of three main components: 
# 
# * **Control** supports the different approaches to control. It includes a single 
# generic class named Controller that includes the thread that runs the systems, iteratively collects the joint actions from the agents, and executes them. The way joint actions are computed differs between the implementation of the sub-classes.
# * **Agents** supports different AI approaches for decision making (e.g., planning, RL etc). It contains an Agent class that is initialized with the agent's DecisionMaker (that represents the decision making procedure that maps states to actions) and the sensor function of the agent that maps the current state of the world to the agent's observation of it. 
# * **Environments** includes an interface class named env_wrapper to the environments in which the agents operate.

# 
# 
# ---
# 
# Let's install the Multi Agent Control (MAC) library as well as the Multi-Taxi environment, on which we will demonstrate the different control approaches. 
# 
# 
# 

# In[1]:


# don't need this with conda env
# !pip install git+https://github.com/sarah-keren/MAC
# !pip install git+https://github.com/sarah-keren/multi_taxi


# Since the focus is on demonstrating the different control dynamics and not on the decision making process, we offer an example in which decision making is random - at each state a random action is selected. We will demonstrate this in a fully centralized and decentralized setting. Let's import the relevant clases. 
# 

# In[6]:


from multi_taxi.env.state import MultiTaxiEnvState
from multi_taxi.utils.types import Event

from src.control.controller_decentralized import DecentralizedController
from src.control.controller_centralized import CentralizedController
from src.agents.agent import Agent, RandomDecisionMaker
from src.decision_makers.Deap_learner import LearningDecisionMaker, SinglePassengerPosWrapper, SingleTaxiWrapper, TaxiObsPrepWrapper
from src.decision_makers.PPO_DM1 import PPODecisionMaker
from src.decision_makers.Short_Term_Planner_DM import ST_Planner
from src.environments.env_wrapper import EnvWrappper, EnvWrappperGym, env_pos_change
import matplotlib.pyplot as plt
from multi_taxi import multi_taxi_v0 as TaxiEnv
from multi_taxi import ObservationType
from multi_taxi import wrappers as WP
from src.decision_makers.MA_decision_makers.MA_AddOn_DM import *

from agents import BCAgent

# some helpful functions
def man_dist(point_a,point_b):
    # calculate manhatten distance from 2 2D points
    return abs(point_a[0]-point_b[0])+abs(point_a[1]-point_b[1])

# check if some other Taxi is close to named taxi - conflict function over state
def cf_close_to_other(state : MultiTaxiEnvState,name,dist = 2,**kwargs):
    t_dict = state.taxi_by_name
    my_pos = t_dict[name].location
    for t in t_dict.keys():
        if t==name: continue
        if man_dist(my_pos,t_dict[t].location)<=dist:
            return True
    return False



# Now, let's create a multi-taxi environment and its MAC wrapper.

# env = TaxiEnv.parallel_env(num_taxis=4, num_passengers=3, pickup_only=True,  observation_type='symbolic', render_mode='human')
env = TaxiEnv.parallel_env(num_taxis=2,
                           num_passengers=2,
                           pickup_only=True,
                           observation_type=ObservationType.MIXED,
                           field_of_view=[None, None],
                           render_mode='human')

# using FIXED places wrappers to set start locations
# env = WP.FixedPassengerStartLocationsWrapper(env, 2, 4, 6, 4)
# env = WP.FixedTaxiStartLocationsWrapper(env, 2,1,2,7)


# # changing reward table for collision
# for k in env.unwrapped.reward_table.values():
#     k[Event.COLLISION] = -50

# reset and render to check
# obs = env.reset()
# env.render()

# making sure env.agents has taxis name
try:
    env = EnvWrappper(env, env.agents)
except:
    env.agents = env.possible_agents
    env = EnvWrappper(env, env.agents)

env.reset()
print('EnvironmentWrapper created')



# ## Decentralized Control
# 
# Let's create a decenrelized contorl setting, in which each agent chooses randomly the action to perform. 
# 
# 

decentralized_agents = {}

# # first we set random DM agents:
# decentralized_agents = {agent_name: Agent(RandomDecisionMaker(env.env.action_spaces[agent_name]))           #can use diffrent DM
#           for agent_name in env.env.agents}

#    # all PPO agents
# decentralized_agents = {agent_name: Agent(PPODecisionMaker(env.env.action_spaces[agent_name],env.env,agent_name))
#                         for agent_name in env.env.agents}


# ---------------------------------------------------------------------------------------------------------
# Conflict_Function - we set a conflict function to help agents to decide whether to use MA DM or its own DM

# ---------------------------------------------------------------------------------------------------------------
# create all agents, using MA_Prune_DM (if cf is on - use MA_DM that prune the worst MA action and return it to
#           it's own DM whitout the pruned action, if cf is off uses its own original DM)
#           stohastic action - if on, add patience feature to decide stohasticly when agent hold still for MA reasons,
#                 as long as it waits more he loses its patience and with higher probability of using its own DM

for agent_name in env.env.agents:
    ag_name = agent_name
    # cf1 = Conflict_Function(None, cf_close_to_other,ag_name,True, man=5, dist=3)
    cf1 = Conflict_Function(func_obs=None, func_state=cf_close_to_other,name=ag_name, is_binary=True, dist=4)
    dm = MA_prune_DM(my_DM=BCAgent(env, ag_name, '2_passenger_bc'),
                     MA_DM=pickup_DM(env.env.action_spaces[ag_name]),
                     conflict_function=cf1,
                     action_space=env.env.action_spaces[ag_name],
                     prune_MA_flag=True,
                     AgentName=ag_name,
                     stohastic_action=True)

    ag = Agent(decision_maker=dm, AgentName=ag_name)
    decentralized_agents[ag_name] = ag


# Here, the action to perform is collected by each agent, we use the known MAC controller

controller = DecentralizedController(env, decentralized_agents)
controller.run(render=True, max_iteration=60)


# %%
