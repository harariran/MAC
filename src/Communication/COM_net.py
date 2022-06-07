# import math
from queue import Queue

# from src.agents.agent import*
# from src.control.Controller_COM import DecentralizedComController
#
# from src.control.controller_decentralized import DecentralizedController





class Message:
    def __init__(self, data, author = None, time_stamp = None, p2p_target = None):
        self.data = data
        self.author = author
        self.time_stamp = time_stamp
        self.size = self.data.__sizeof__()           #len(data)


class COM_net:

    def __init__(self, Delivery_target_type : str = "All", Delivert_latency : int = 0 , Noise_Net = None, Team_group = None ):
        self.Delivery_target_type = Delivery_target_type
        self.Delivert_latency = Delivert_latency
        self.delivery_buffer = Queue()
        for i in range(Delivert_latency):
            self.delivery_buffer.put(None)
        self.Noise_net = Noise_Net
        self.Team = Team_group


    def update_delievery(self, Joint_messages : dict, Agents):
        self.delivery_buffer.put(Joint_messages)
        Out_Joint_message = self.delivery_buffer.get()
        if self.Delivery_target_type == "All":
            # all_messages = [message.value for message in Out_Joint_message]
            return {agent : Out_Joint_message for agent in Agents}
        elif self.Team != None:
            Team_messages = {agent : Out_Joint_message[agent] for agent in self.Team}
            all_messages = [message for message in Team_messages]
            return {agent : all_messages for agent in Agents}
        else: return {}









"""
An abstract class for choosing an action, part of an agent.
(An agent can have one or several of these)
"""
class DecisionMaker:

    def __init__(self):
        pass

    def get_action(self, observation):
        pass

    """
    Functions for training:
    """
    def get_train_action(self, observation):
        pass

    def update_step(self, obs, action,new_obs, reward, done):
        pass

    def update_episode(self, batch_size=0):
        pass

class RandomDecisionMaker:
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):
        if type(self.space) == dict:
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()




