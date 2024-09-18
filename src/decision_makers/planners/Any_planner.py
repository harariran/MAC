import time
from IPython.display import clear_output

from IPython.core.display_functions import clear_output
from ai_dm.Environments.gymnasium_envs.gymnasium_problem import GymnasiumProblemS
from ai_dm.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search, a_star, depth_first_search_l
import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic
from multi_taxi import multi_taxi_v0 as TaxiEnv, Action
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


planning_par_env = TaxiEnv.parallel_env(
    num_taxis=1,
    num_passengers=2,
    pickup_only=True,
    render_mode='human'
)
planning_par_env.reset()
mt_problem = MultiTaxiProblem(planning_par_env, planning_par_env.state())
planning_par_env.render()

# # the function only accepts dicitonary joint actions
# action = {'taxi_0': planning_par_env.unwrapped.get_action_map('taxi_0')[Action.NORTH.value]}
#
# # get transition info from current state
# transitions = planning_par_env.unwrapped.state_action_transitions(planning_par_env.state(), action)
#
# print(f'num transitions {len(transitions)} - expected 1 because actions are deterministic')
# for i, (new_state, rewards, terms, truncs, infos, prob) in enumerate(transitions, 1):
#     print(f'transision {i} rewards: {rewards["taxi_0"]}')
#     print(f'transision {i} terms: {terms["taxi_0"]}')
#     print(f'transision {i} infos: {infos["taxi_0"]}')
#
# planning_par_env.unwrapped.set_state(new_state)
# planning_par_env.render()



# get solution from BFS algorithm
print('making plan...')
sol_len, final_node, solution, explore_count, terminated = breadth_first_search(mt_problem)
solution = [eval(action) for action in solution]  # returns dict strings (for some reason???) fix with `eval`

sol = [a['taxi_0'] for a in solution]

print([planning_par_env.unwrapped.get_action_meanings('taxi_0')[a] for a in sol])

# execute solution
while True:
    if not solution:  # check solution complete without done
        print('failure')
        break

    # parallel API gets next observations, rewards, terms, truncs, and infos upon `step`
    # all values are dictionaries
    observations, rewards, terms, truncs, infos = planning_par_env.step(solution.pop(0))

    # re-render after step
    time.sleep(0.15)  # sleep for animation speed control
    clear_output(wait=True)
    planning_par_env.render()  # clear previous render for animation effect

    if all(terms.values()):  # check dones
        print('success!')
        break
    if all(truncs.values()):
        print('truncated')
        break


