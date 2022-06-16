import math

from src.Communication.COM_net import COM_net
from src.agents.agent import DecisionMaker, Action_message_agent, Agent_Com
from src.control.Controller_COM import DecentralizedComController
from src.decision_makers.planners.MA_com_planner import Astar_message_DM
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

TAXI_pickup_dropoff_REWARDS = dict(
    step=-1,
    no_fuel=-1,
    bad_pickup=-1,
    bad_dropoff=-1,
    bad_refuel=-1,
    bad_fuel=-1,
    pickup=100,
    standby_engine_off=0,
    turn_engine_on=-1,
    turn_engine_off=-1,
    standby_engine_on=0,
    intermediate_dropoff=100,
    final_dropoff=100,
    hit_wall=-1,
    collision=-1,
    collided=-1,
    unrelated_action=-1,
)

"""
Builds Multi_taxi env
"""
m = MAP
env = MultiTaxiEnv(num_taxis=3, num_passengers=5, domain_map=m, observation_type='symbolic',rewards_table=TAXI_pickup_dropoff_REWARDS ,option_to_stand_by=True)

# env = SingleTaxiWrapper(env)
# obs = env.reset()

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
    - implement Agent_com class (inherit from Agent_com
    - make sure to implement set_data_func - that decides what is the data that the agent will transmit whenever it called
    - u can implement your recieve_func - that decides what to do with a recieved data
        defualt is the union_func that add the message data to the observation 
"""

class Heading_message_agent(Agent_Com):

    def __init__(self, decision_maker : Astar_message_DM , sensor_function =None, message_filter = None, AgentName = None, bandW = math.inf, union_recieve = False):
        super().__init__(decision_maker , sensor_function, message_filter, AgentName, bandW, union_recieve)
        self.last_action = None
        self.last_message = None

    def set_data_func(self, obs):
        data = (self.decision_maker.taking_passenger,len(self.decision_maker.active_plan))
        return data

    # implement our recive_func
    def set_recive_func(self, obs, message):
        self.last_message = message
        self.decision_maker.save_last_message(message)
        # self.decision_maker.updateplan_message(message)

    # saves last action of the agent - not necessary for com module
    def set_last_action(self, action):
        self.last_action = action
#
#


# after having our com-Astar-agents class we can set our env agents into a dicentralized_agents dict
env_agents = environment.get_env_agents()
decentralized_agents = {agent_name: Heading_message_agent(Astar_message_DM(env ,single_plan=True, Taxi_index=int(agent_name[-1]), domain_map=m) ,AgentName=agent_name)             # Agent(LearningDecisionMaker(env.action_space))  # can use diffrent DM
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
#communicate first
controller.send_recieve()

#run (communication inside after each time_click)
controller.run(render=True, max_iteration=200,reset=True)
print("Thats all")

