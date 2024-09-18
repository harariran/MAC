from ai_dm.Environments.gym_envs.gym_problem import GymProblem
from ai_dm.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search, a_star, depth_first_search_l
import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic
from multi_taxi import multi_taxi_v0 as TaxiEnv
from multi_taxi import ObservationType
from itertools import product

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

        # HERE WE USE OUR TRANSITION FUNCTION
        transitions = self.env.unwrapped.state_action_transitions(node.state.key, action)

        action_cost = self.get_action_cost(action, node.state)
        for next_state, rewards, terms, truncs, infos, prob in transitions:

            info={}
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

# reset environment and create the problem object for it
planning_par_env.reset()
mt_problem = MultiTaxiProblem(planning_par_env, planning_par_env.state())



# import gym
#
# taxi_env = gym.make("Taxi-v3").env
#
# taxi_env.reset ()
# taxi_env.render()

def create_env():

    # define the environment
    taxi_env = env = TaxiEnv.parallel_env(num_taxis=1,
                           num_passengers=2,
                           pickup_only=False,
                           observation_type=ObservationType.MIXED,
                           field_of_view=[None],
                           render_mode='human')
    taxi_env.reset()
    # init_state = taxi_env.encode(0, 1, 0, 1)  # (taxi row, taxi column, passenger index, destination index)
    # taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    # print(taxi_row)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    taxi_env.render()
    return taxi_env

taxi_env = create_env()

# create a wrapper of the environment to the search
taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)


# perform BFS
[best_value, best_node, best_plan, explored_count, ex_terminated] = breadth_first_search(problem=taxi_p,
                                                                                         log=True,
                                                                                         log_file=None,
                                                                                         iter_limit=defs.NA,
                                                                                         time_limit=defs.NA,
                                                                                        )

print(best_plan)
for action_id in best_plan:
    taxi_p.apply_action(action_id)
    taxi_p.env.render()


# from
# gym_problem.
# from itertools import product
#
#
# class MultiTaxiProblem(GymnasiumProblemS):
#     def sample_applicable_actions_at_state(self, state, sample_size=None):
#         action_lists = []
#         for agent in self.env.possible_agents:
#             action_lists.append(self.env.unwrapped.get_action_meanings(agent).keys())
#
#         # get list of possible joint actions
#         possible_joint_actions_tuples = list(product(*action_lists))
#         return [{self.env.possible_agents[i]: action
#                  for i, action in enumerate(joint_action)}
#                 for joint_action in possible_joint_actions_tuples]
#
#     def get_action_cost(self, action, state):
#         return 1
#
#     def get_successors(self, action, node):
#         successor_nodes = []
#
#         # HERE WE USE OUR TRANSITION FUNCTION
#         transitions = self.env.unwrapped.state_action_transitions(node.state.key, action)
#
#         action_cost = self.get_action_cost(action, node.state)
#         for next_state, rewards, terms, truncs, infos, prob in transitions:
#             info = {}
#             info['prob'] = prob
#             info['reward'] = rewards
#             info.update(infos)
#
#             # state is a hashable key
#             successor_state = utils.State(key=next_state, is_terminal=all(terms.values()))
#
#             successor_node = utils.Node(state=successor_state,
#                                         parent=node,
#                                         action=action,
#                                         path_cost=node.path_cost + action_cost,
#                                         info=info)
#
#             successor_nodes.append(successor_node)
#
#         return successor_nodes
#
#
# # reset environment and create the problem object for it
# planning_par_env.reset()
# mt_problem = MultiTaxiProblem(planning_par_env, planning_par_env.state())