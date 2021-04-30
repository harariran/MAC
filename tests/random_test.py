import argparse
import sys

import numpy as np

sys.path.append('..')
from control.controls import CentralizedControl, DecentralizedControl
from utils.random_agent import RandomAgent

EMPTY_OBSERVATION_FILTER = lambda x: x

def main():
    args = parse_args()
    env = set_env(args.env)
    centralized_random_test(env)
    decentralized_random_test(env)

def centralized_random_test(env):
    agent = RandomAgent(env.action_space)
    observation = env.reset()
    controller = CentralizedControl(env, agent)
    controller.run(100)

def decentralized_random_test(env):
    agents = [RandomAgent(space) for space in env.action_space]
    observation = env.reset()
    controller = DecentralizedControl(env, agents)
    controller.run(100)

def set_env(environment_name):
    print('Initializing environment...')
    
    if environment_name == 'taxi':
        sys.path.append('../environments/MultiTaxiEnv')
        from MultiTaxiWrapper import MultiTaxiWrapper
        num_taxis = 2
        env = MultiTaxiWrapper(2, 1)

    elif environment_name == 'particle':
        sys.path.append('../environments/multiagent-particle-envs')
        import make_env
        env = make_env.make_env('simple_spread')
        env.discrete_action_input = True

    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--env',
        required=True,
        choices=['taxi', 'particle'],
        help='Environment to run test on.'
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()