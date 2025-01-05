import random
import sys
from abc import ABC, abstractmethod

from multi_taxi.env.state import MultiTaxiEnvState

from src.agents.agent import Agent, DecisionMaker
from src.decision_makers.Short_Term_Planner_DM import ST_Planner


def man_dist(point_a,point_b):
    # calculate manhatten distance from 2 2D points
    return abs(point_a[0]-point_b[0])+abs(point_a[1]-point_b[1])

# class Conflict_Function - accepts observation or a state and returns :
    # if binary - 0/1 (0 - no conflict, 1 - conflict)
    # if non_binary - [0-1] (span from 0 - no conflict to 1 - conflict)
class Conflict_Function:
    def __init__(self,  func_obs = None , func_state = None, name = None, is_binary= True, *argv, **kwargs):
        # ...
        self.my_obs_function = func_obs
        self.my_state_function = func_state
        self.is_binary = is_binary
        self.my_name = name
        self.my_args = argv
        self.my_kwargs = kwargs
        # ...

    def cf_observation(self, obs):
        # ...
        return self.my_obs_function(obs)
        # ...

    def cf_state(self, state):
        # ...
        return self.my_state_function(state,self.my_name, *self.my_args, **self.my_kwargs)
        # ...

# gets random conflict
def random_func(*args):
    return random.randint(0,1)
random_cf = Conflict_Function(func_obs=random_func, func_state=random_func)



class MA_AddOn_DM(DecisionMaker, ABC):

    def __init__(self, action_space = None, AgentName = None, conflict_function : Conflict_Function = random_cf, my_DM : DecisionMaker= None, MA_DM : DecisionMaker = None, patience_factor = 0.5):
        self.conflict_function = conflict_function
        self.agent_name = AgentName
        self.my_DM = my_DM
        self.MA_DM = MA_DM
        self.sum_conflicts = 0
        self.patience_factor_lambda = patience_factor
        self.stand_still_counter = 0
        self.last_dm = my_DM


    def get_action(self, observation, state=None):
        if state:
            cf = self.conflict_function.cf_state(state)
        else:
            cf = self.conflict_function.cf_observation(observation)
        self.sum_conflicts += cf
        if cf:
            self.last_dm = self.MA_DM
            if isinstance(self.my_DM,ST_Planner):
                self.my_DM.del_plan()
        else:
            self.last_dm = self.my_DM
        return self.last_dm.get_action(observation)

    def is_conflict_on(self):
        return self.MA_DM==self.last_dm


# TODO - complete MA DM that can get pruned action and still make desition
class pickup_DM(DecisionMaker, ABC):

    def __init__(self, action_space=None, AgentName=None):
        self.agent_name = AgentName
        self.action_space = action_space

    def get_action(self, observation, state=None , pruned_action = None):
        return 4 # TODO

    def get_action_pruned(self, observation, state=None):
        pass  # TODO


def closest_direction(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > abs(dy):
        if dx > 0:
            return 2  # East
        else:
            return 3  # West
    else:
        if dy > 0:
            return 1  # North
        else:
            return 0  # South

# check if some other Taxi is close to named taxi - conflict function over state
def prune_action(state : MultiTaxiEnvState,name, **kwargs):
    t_dict = state.taxi_by_name
    my_pos = t_dict[name].location
    min_dist = sys.maxsize
    for t in t_dict.keys():
        if t==name: continue
        if man_dist(my_pos,t_dict[t].location)<min_dist:
            closest_pos = t_dict[t].location

    return closest_direction(my_pos[1],-my_pos[0], closest_pos[1], -closest_pos[0])


import math


def shifted_exponential(x: int, lambda_  : float = 0.2) -> float:
    """
    Calculate the shifted exponential distribution value for a given integer x and lambda.

    Parameters:
    x (int): The input integer value.
    lambda_ (float): The rate parameter (lambda) of the exponential distribution.

    Returns:
    float: The shifted exponential distribution value, which is between 0 and 1.
    """
    # Ensure lambda is positive
    if lambda_ <= 0:
        raise ValueError("Lambda must be greater than 0.")

    # Compute the CDF of the exponential distribution
    return 1 - math.exp(-lambda_ * (x))

# checks if there is state or obs
def state_or_obs (state, obs):
    if state:
        return state
    else:
        return obs


def is_equal(instance1, instance2) -> bool:
    """
    Check if all attributes of two instances of the same class are equal.

    Parameters:
    instance1 (object): The first instance to compare.
    instance2 (object): The second instance to compare.

    Returns:
    bool: True if all attributes of both instances are the same, False otherwise.
    """
    # Ensure both instances are of the same class
    if instance1.__class__ != instance2.__class__:
        raise ValueError("Both instances must be of the same class.")

    # Get attributes of both instances
    attrs1 = vars(instance1)  # or instance1.__dict__
    attrs2 = vars(instance2)  # or instance2.__dict__

    # Check if the attributes are the same
    return attrs1 == attrs2


def stand_still(name,state_obs_a, state_obs_b):
    try:
        a = state_obs_a.taxi_by_name[name]
        b = state_obs_b.taxi_by_name[name]
        a1 = str( [a.location,len(a.passengers)])
        b1 = str([b.location,len(b.passengers)])
        if a1==b1:
            return True
        else:
            return False
    except:
        return False

class MA_prune_DM(DecisionMaker, ABC):

    def __init__(self, action_space = None, AgentName = None, conflict_function : Conflict_Function = random_cf, my_DM : DecisionMaker= None, MA_DM : DecisionMaker = None, prune_MA_flag = False,  patience_factor = 0.2, stohastic_action = False):
        self.agent_name = AgentName
        self.my_DM = my_DM
        self.MA_DM = MA_DM
        self.last_dm = my_DM

        self.conflict_function = conflict_function
        self.sum_conflicts = 0
        self.conflict_flag = False

        self.prune_MA_flag = prune_MA_flag
        self.last_pruned_action = None

        # variables for stohastic action according to patience
        self.stohastic_action = stohastic_action
        self.patience_factor_lambda = patience_factor
        self.stand_still_counter = 0
        self.last_state = None




    def get_action(self, observation, state=None):
        # check if a state or observation
        st_obs = state_or_obs(state, observation)
        if state:
            self.conflict_flag = self.conflict_function.cf_state(state)
        else:
            self.conflict_flag = self.conflict_function.cf_observation(observation)
        if self.conflict_flag: self.sum_conflicts += 1

        # if there was a conflict - use MA idea
        if self.conflict_flag:
            if self.stohastic_action:
                if stand_still(self.agent_name,st_obs, self.last_state):
                    self.stand_still_counter+=1
                else:
                    self.stand_still_counter=0
                self.last_state = st_obs
            self.last_dm = self.MA_DM
            if self.prune_MA_flag:
                pruned_a = prune_action(state, self.agent_name)
                self.last_pruned_action =pruned_a
                if self.stohastic_action:
                    prob_greedy = shifted_exponential(self.stand_still_counter, self.patience_factor_lambda)
                    # TODO hide
                    print("stohastic action, patience chances to act greeady (not MA) :" + str(prob_greedy))
                    # case greeady and choosing own pie
                    if random.random() < prob_greedy:
                        self.last_dm = self.my_DM
                        a = self.my_DM.get_action(observation)
                    # case choosing MA pie
                    else:
                        self.last_dm = self.MA_DM
                        a = self.my_DM.get_action(observation, pruned_action=pruned_a)
                return a
            else:
                if isinstance(self.my_DM,ST_Planner):
                    self.my_DM.del_plan()
                return self.MA_DM.get_action(observation)
        else:
            if self.stohastic_action:
                self.stand_still_counter = 0
                self.last_state = st_obs
            self.last_dm = self.my_DM
            return self.my_DM.get_action(observation)

    def is_conflict_on(self):
        return self.conflict_flag


if __name__ == '__main__':
    # check code

    # def check_func(a):
    #     if a > 5:
    #         return 0
    #     else:
    #         return 1
    #
    #
    # cf1 = Conflict_Function(func_obs= check_func, name='taxi_0', is_binary=True)
    # ans = cf1.cf_observation(4)
    #
    # cf2 = random_cf
    # ans = cf2.cf_observation(4)
    # print(ans)
    # for i in range(0,6):
    #     ans = cf2.cf_observation(4)
    #     print(ans)
    #     ans = cf2.cf_state(7)
    #     print(ans)

    # Example usage:
    x_value = 0
    lambda_value = 0.20
    while x_value < 10 :

        print(x_value)
        result = shifted_exponential(x_value, lambda_value)
        x_value += 1
        print(f"The shifted exponential distribution value is: {result} \n")

