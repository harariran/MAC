

from src.decision_makers.MA_decision_makers.MADDPG import MADDPG

from multi_taxi import multi_taxi_v0, ObservationType, FuelType
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os



def create_env(render=False):
    render_mode = 'human' if render else None
    env = multi_taxi_v0.parallel_env(
        num_taxis=2,
        num_passengers=2,
        pickup_only=True,
        observation_type=ObservationType.SYMBOLIC,
        max_steps=100
    )
    return env

def preprocess_observation(obs):
    return np.array(obs, dtype=np.float32)


def train(maddpg, n_episodes=10000, evaluate_every=100):
    env = create_env()
    episode_rewards = []

    for episode in range(n_episodes):
        observations = env.reset()
        episode_reward = np.zeros(len(env.agents))

        # Get states for all agents in ordered list
        agent_list = list(env.agents)  # Fix order of agents
        states = [preprocess_observation(observations[agent_id]) for agent_id in agent_list]

        while env.agents:
            # Get actions
            actions = {}
            actions_list = []
            for idx, agent_id in enumerate(agent_list):
                action = maddpg.agents[idx].choose_action(states[idx])
                actions[agent_id] = int(action)
                actions_list.append(action)

            # Environment step
            next_observations, rewards, terms, truncs, _ = env.step(actions)

            # Process next states in same order
            next_states = [preprocess_observation(next_observations[agent_id]) for agent_id in agent_list]

            # Get rewards and dones in same order
            rewards_list = [rewards[agent_id] for agent_id in agent_list]
            dones_list = [terms[agent_id] or truncs[agent_id] for agent_id in agent_list]

            maddpg.memory.store_transition(states, actions_list, rewards_list, next_states, dones_list)

            if maddpg.memory.mem_cntr > maddpg.batch_size:
                maddpg.learn()

            states = next_states
            for idx, agent_id in enumerate(agent_list):
                episode_reward[idx] += rewards[agent_id]

            if all(terms.values()) or all(truncs.values()):
                break

        mean_reward = np.mean(episode_reward)
        episode_rewards.append(mean_reward)

        if episode % evaluate_every == 0:
            print(f"Episode {episode}: Reward: {mean_reward:.2f}")
            maddpg.save_models(f"models/episode_{episode}")

    return episode_rewards

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    env = create_env()
    obs = env.reset()
    first_agent = env.agents[0]

    sample_obs = preprocess_observation(obs[first_agent])
    state_dim = len(sample_obs)
    n_actions = env.action_spaces[first_agent].n

    # Initialize MADDPG
    maddpg = MADDPG(
        n_agents=len(env.agents),
        state_dim=state_dim,
        n_actions=n_actions,
        buffer_size=100000,
        batch_size=64
    )

    rewards = train(maddpg, n_episodes=10000)
    rewards = train(maddpg, n_episodes=10000)