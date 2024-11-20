import numpy

from src.agents.agent import Agent, DecisionMaker, RandomDecisionMaker
from src.environments.env_wrapper import*

from multi_taxi.world.entities import Taxi
from multi_taxi import multi_taxi_v0 as TaxiEnv, Action
from multi_taxi import ObservationType
from itertools import product


from ai_dm.Environments.gymnasium_envs.gymnasium_problem import GymnasiumProblemS
from ai_dm.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search, a_star, depth_first_search_l
import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic

# copy Taxi_b parameters to Taxi_a (without the name)
def set_same_taxi_values(Taxi_a:Taxi, Taxi_b:Taxi):
    Taxi_a.location = Taxi_b.location  # assert tuple type for immutability
    Taxi_a.max_capacity = Taxi_b.max_capacity
    Taxi_a.fuel = Taxi_b.fuel
    Taxi_a.max_fuel = Taxi_b.max_fuel
    Taxi_a.fuel_type = Taxi_b.fuel_type
    Taxi_a.n_steps = Taxi_b.n_steps
    Taxi_a.max_steps = Taxi_b.max_steps
    Taxi_a.passengers = Taxi_b.passengers
    Taxi_a.can_collide = Taxi_b.can_collide
    Taxi_a.collided = Taxi_b.collided
    Taxi_a.engine_on = Taxi_b.engine_on
    # set taxi color for rendering
    Taxi_a.color = Taxi_b.color

class MultiTaxiProblem(GymnasiumProblemS):
    def sample_applicable_actions_at_state(self, state, sample_size=None):
        action_lists = []
        for agent in self.env.possible_agents:
            action_lists.append(self.env.unwrapped.get_action_meanings(agent).keys())

        # get list of possible joint actions
        possible_joint_actions_tuples = list(product(*action_lists))
        return [{self.env.possible_agents[i]: action
                 for i, action in enumerate(joint_action)}
                for joint_action in possible_joint_actions_tuples]

    def get_action_cost(self, action, state):
        return 1

    def get_successors(self, action, node):

        successor_nodes = []
        a = action.get('taxi_0')
        # prune the fisrt action only
        if isinstance(self.env, Pruned_action_env) and (a == self.env.pruned_action) and not self.env.pruned :
            self.env.pruned=True
            return successor_nodes

        # HERE WE USE OUR TRANSITION FUNCTION
        transitions = self.env.unwrapped.state_action_transitions(node.state.key, action)

        action_cost = self.get_action_cost(action, node.state)
        for next_state, rewards, terms, truncs, infos, prob in transitions:
            info = {}
            info['prob'] = prob
            info['reward'] = rewards
            info.update(infos)

            # state is a hashable key
            successor_state = utils.State(key=next_state, is_terminal=all(terms.values()))

            successor_node = utils.Node(state=successor_state,
                                        parent=node,
                                        action=action,
                                        path_cost=node.path_cost + action_cost,
                                        info=info)

            successor_nodes.append(successor_node)

        return successor_nodes

# fix too many wrappers env
def fix_env_for_render(env):
    temp_env = env.env
    while True :
        temp_env = temp_env.env
        if isinstance(temp_env, TaxiEnv.MultiTaxiEnv) :
            break
    return temp_env

# Makes a single agent texi env from a multi
def create_single_env(ma_env, agent_name = 'taxi_0', render = False):
    if ma_env.unwrapped.num_taxis == 1:
        return ma_env.copy()
    else:
        temp_env = TaxiEnv.parallel_env(
            num_taxis=1,
            num_passengers=ma_env.unwrapped.num_passengers,
            pickup_only=ma_env.unwrapped.pickup_only,
            observation_type=ma_env.unwrapped.observation_type,
            field_of_view=ma_env.unwrapped.field_of_view,
            render_mode=ma_env.unwrapped.render_mode)
        temp_env.reset()
        st = ma_env.unwrapped.state()
        s = temp_env.unwrapped.state()
        set_same_taxi_values(s.taxi_by_name['taxi_0'],st.taxi_by_name[agent_name])
        # s.taxi_by_name['taxi_0'] = st.taxi_by_name[agent_name]
        # s.taxis[0] = st.taxi_by_name[agent_name]
        # temp_env.unwrapped.set_locations(loc)
        # n = s.taxi_by_name['taxi_0'].name

        s.passengers= st.passengers
        temp_env.unwrapped.set_state(s)
        new_s = temp_env.unwrapped.state()
        if isinstance(ma_env,Pruned_action_env):
            temp_env= Pruned_action_env(temp_env, ma_env.pruned_action)

    # temp_env.observation_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
    # temp_env.action_space = temp_env.action_spaces[temp_env.possible_agents[0]]
    # temp_env.action_space = Discrete(temp_env.action_space.n)

    if render:
        try:
            ma_env.render()
        except:
            ma_temp_render_env = fix_env_for_render(ma_env)
            ma_temp_render_env.render()
        temp_env.env.render()
    return temp_env



class ST_Planner(DecisionMaker):
    def __init__(self, action_space, ma_env, agent_name = 'taxi_0', render = False):
        self.space = action_space
        self.ma_env = ma_env
        self.agent_name = agent_name
        self.render = render
        # self.demo_env = copy.deepcopy(env)
        try:
            self.init_state = copy.deepcopy(ma_env.state())
        except:
            print('env has not ben reset yet, reset() now has been called')
            ma_env.reset()
            self.init_state = copy.deepcopy(ma_env.state())
        self.single_problem = create_single_env(ma_env ,agent_name, render= self.render)
        self.plan = []

               # for debug:
        # print(f"{self.model.observation_space}")
        # print(f"model {self.model.obs_meanings}")
        # print(f"env {self.env.unwrapped.get_observation_meanings(agent_name)}")


        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

    def plan_bfs(self):
        mt_problem = MultiTaxiProblem(self.single_problem, self.single_problem.state())
        if self.render:
            print('rendering problem:')
            self.single_problem.render()
        print('making plan using bfs...')
        self.sol = sol_len, final_node, solution, explore_count, terminated = breadth_first_search(mt_problem)
        solution = [eval(action) for action in solution]  # returns dict strings (for some reason???) fix with `eval`
        self.plan = [a['taxi_0'] for a in solution]
        print([self.single_problem.unwrapped.get_action_meanings('taxi_0')[a] for a in self.plan])

    def replan(self):
        self.single_problem = create_single_env(self.ma_env ,self.agent_name, render=self.render)
        self.plan = []
        self.plan_bfs()

    def del_plan(self):
        self.plan = []


    # def create_single_env(self, ma_env):
    #     if ma_env.unwrapped.num_taxis == 1:
    #         return ma_env.copy()
    #     else:
    #         temp_env = SingleTaxi.TaxiEnv.parallel_env(
    #             num_taxis=1,
    #             num_passengers=self.ma_env.unwrapped.num_passengers,
    #             pickup_only=self.ma_env.unwrapped.pickup_only,
    #             observation_type=self.ma_env.unwrapped.observation_type,
    #             field_of_view=self.ma_env.unwrapped.field_of_view,
    #             render_mode=self.ma_env.unwrapped.render_mode)
    #     try:
    #         self.obs_meanings = temp_env.unwrapped.get_observation_meanings()
    #     except:
    #         self.obs_meanings = temp_env.unwrapped.get_observation_meanings('taxi_0')
    #
    #     temp_env.reset()
    #     # temp_env.observation_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
    #     # temp_env.action_space = temp_env.action_spaces[temp_env.possible_agents[0]]
    #     # temp_env.action_space = Discrete(temp_env.action_space.n)
    #     temp_env.env.render()




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

    def get_action(self, observation=None, pruned_action = None):

        if (pruned_action != None):
            self.ma_env = Pruned_action_env(env=self.ma_env, pruned_action=pruned_action)
        if (len(self.plan)==0) or self.plan[0]==pruned_action:
            self.replan()
        action = self.plan.pop(0)
        return action


# check code
#--------------------------

# MA_env = TaxiEnv.parallel_env(
#     num_taxis=2,
#     num_passengers=2,
#     pickup_only=True,
#     render_mode='human'
# )
# MA_env.reset()
# random_name = 'taxi_0'
# plan_name = 'taxi_1'
# ag2_DM = ST_Planner(MA_env.action_space(plan_name),MA_env, plan_name)
# ag1_DM = RandomDecisionMaker(MA_env.env.action_spaces[random_name])
#
#
# dms = {random_name: ag1_DM, plan_name: ag2_DM}
# # execute solution
#
# MA_env.env.render()
# while True:
#
#     # arange next action as a joint action to be executed in parallel for all agents
#     joint_action = {agent: dms[agent].get_action() for agent in MA_env.agents}
#
#     # parallel API gets next observations, rewards, terms, truncs, and infos upon `step`
#     # all values are dictionaries
#
#     observations, rewards, terms, truncs, infos = MA_env.step(joint_action)
#
#     # re-render after step
#     # time.sleep(0.15)  # sleep for animation speed control
#     # clear_output(wait=True)
#     MA_env.render()  # clear previous render for animation effect
#
#     if all(terms.values()):  # check dones
#         print('success!')
#         break
#     if all(truncs.values()):
#         print('truncated')
#         break