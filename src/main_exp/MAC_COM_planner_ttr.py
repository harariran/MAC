import math

from src.Communication.COM_net import COM_net
from src.agents.agent import DecisionMaker, Action_message_agent, Agent_Com
from src.control.Controller_COM import DecentralizedComController
from src.decision_makers.planners.map_planner import AstarDM
from src.environments.env_wrapper import EnvWrappper
from multi_taxi import MultiTaxiEnv


MAP2 = [
    "+-------+",
    "| : |F: |",
    "| : | : |",
    "| : : : |",
    "| | :G| |",
    "+-------+",
]

MAP = [
    "+-----------------------+",
    "| : |F: | : | : | : |F: |",
    "| : | : : : | : | : | : |",
    "| : : : : : : : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : : : : : : : |",
    "| | :G| | | :G| | | : | |",
    "+-----------------------+",
]

"""
Builds Multi_taxi env
"""
env = MultiTaxiEnv(num_taxis=2, num_passengers=2, domain_map=MAP, observation_type='symbolic')

# env = SingleTaxiWrapper(env)
obs = env.reset()

# env.render()

# # Make sure it works with our API:
env.agents = env.taxis_names
# print(f"{env.agents}\n")
env.action_spaces = {
    agent_name: env.action_space for agent_name in env.agents
}
env.observation_spaces = {
    agent_name: env.observation_space for agent_name in env.agents
}
env.possible_agents = [agent for agent in env.agents]
#
# # env = SingleTaxiWrapper(env)
# # env = SinglePassengerPosWrapper(environment, taxi_pos=[0, 0])
environment = EnvWrappper(env, env.agents)
#
print('EnvironmentWrapper created')

# making agent class that communicates and heads towards 1 passenger (pickup->dropoff)
"""
in order to use com module: 
    - implement Agent_com class (inherit from Agent_com)
        - make sure to implement set_data_func - that decides what is the data that the agent will transmit whenever it called
        - u can implement your recieve_func - that decides what to do with a recieved data
            defualt is the union_func that add the message data to the observation 
    - use DecentralizedComController to execute your com module with all your Agent_com agents in your environment and
"""

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker : AstarDM , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = True):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None

    def set_data_func(self, obs):
        data = (self.decision_maker.taking_passenger,len(self.decision_maker.active_plan))
        return data

    # todo - implement your recive_func
    def set_recive_func(self, obs, message):
        pass

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action
#
#


# after having our com-Astar-agents class we can set our env agents into a dicentralized_agents dict
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(AstarDM(env ,single_plan=True, Taxi_index=int(agent_name[-1]), domain_map=MAP) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
                        for agent_name in env_agents}


"""
Simple use of communication network - 
    - build one using COM_net() - defualt architecture is - all masseges sent to all other agents
            *U can use more options - see at COM_net() class doc.
"""
com = COM_net()
""" 
    - initailze our new controller (DecentralizedComController) - using our env, our agents and our com module
        - this controller will perform all joint action and message delieveries at any time-step
"""
controller = DecentralizedComController(environment, decentralized_agents, com)
"""
activate 
"""
controller.run(render=True, max_iteration=50)
print("Thats all")

