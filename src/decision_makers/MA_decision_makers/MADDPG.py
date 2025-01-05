import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])


class ActorNetwork(tf.keras.Model):
    def __init__(self, n_actions, state_dim, fc1_dims=64, fc2_dims=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.fc3 = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)


class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, fc1_dims=64, fc2_dims=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        state, action = inputs
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q(x)


class MultiAgentReplayBuffer:
    def __init__(self, max_size, n_agents, state_dims, action_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.action_dims = action_dims  # Add this line to store action dimensions

        self.state_memory = [np.zeros((max_size, state_dims)) for _ in range(n_agents)]
        self.new_state_memory = [np.zeros((max_size, state_dims)) for _ in range(n_agents)]
        self.action_memory = [np.zeros(max_size, dtype=np.int32) for _ in range(n_agents)]
        self.reward_memory = [np.zeros(max_size) for _ in range(n_agents)]
        self.terminal_memory = [np.zeros(max_size, dtype=np.bool_) for _ in range(n_agents)]

    def store_transition(self, states, actions, rewards, next_states, dones):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.state_memory[agent_idx][index] = states[agent_idx]
            self.new_state_memory[agent_idx][index] = next_states[agent_idx]
            self.action_memory[agent_idx][index] = actions[agent_idx]
            self.reward_memory[agent_idx][index] = rewards[agent_idx]
            self.terminal_memory[agent_idx][index] = dones[agent_idx]

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = [self.state_memory[i][batch] for i in range(self.n_agents)]
        actions = [np.eye(self.action_dims)[self.action_memory[i][batch]] for i in range(self.n_agents)]
        rewards = [self.reward_memory[i][batch] for i in range(self.n_agents)]
        next_states = [self.new_state_memory[i][batch] for i in range(self.n_agents)]
        dones = [self.terminal_memory[i][batch] for i in range(self.n_agents)]

        return Experience(states, actions, rewards, next_states, dones)


class MADDPG:
    def __init__(self, n_agents, state_dim, n_actions, buffer_size=100000, batch_size=64,
                 gamma=0.99, alpha=0.001, beta=0.002, tau=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.agents = []
        for i in range(n_agents):
            agent = MADDPGAgent(i, state_dim, n_actions, n_agents, alpha, beta, gamma, tau)
            self.agents.append(agent)

        self.memory = MultiAgentReplayBuffer(buffer_size, n_agents, state_dim, n_actions)

    def save_models(self, path):
        for idx, agent in enumerate(self.agents):
            agent_path = f"{path}/agent_{idx}"
            agent.actor.save(f"{agent_path}_actor")
            agent.critic.save(f"{agent_path}_critic")
            agent.target_actor.save(f"{agent_path}_target_actor")
            agent.target_critic.save(f"{agent_path}_target_critic")

    def load_models(self, path):
        for idx, agent in enumerate(self.agents):
            agent_path = f"{path}/agent_{idx}"
            agent.actor = tf.keras.models.load_model(f"{agent_path}_actor")
            agent.critic = tf.keras.models.load_model(f"{agent_path}_critic")
            agent.target_actor = tf.keras.models.load_model(f"{agent_path}_target_actor")
            agent.target_critic = tf.keras.models.load_model(f"{agent_path}_target_critic")

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        states = [tf.convert_to_tensor(state, dtype=tf.float32) for state in states]
        actions = [tf.convert_to_tensor(action, dtype=tf.float32) for action in actions]
        rewards = [tf.convert_to_tensor(reward, dtype=tf.float32) for reward in rewards]
        next_states = [tf.convert_to_tensor(next_state, dtype=tf.float32) for next_state in next_states]
        dones = [tf.convert_to_tensor(done, dtype=tf.float32) for done in dones]

        for agent_idx, agent in enumerate(self.agents):
            with tf.GradientTape() as tape:
                next_actions = [a.target_actor(next_states[i])
                                for i, a in enumerate(self.agents)]
                next_actions_concat = tf.concat(next_actions, axis=1)

                target_critic_value = tf.squeeze(agent.target_critic(
                    (next_states[agent_idx], next_actions_concat)), 1)

                current_actions_concat = tf.concat(actions, axis=1)
                critic_value = tf.squeeze(agent.critic(
                    (states[agent_idx], current_actions_concat)), 1)

                target = rewards[agent_idx] + self.gamma * target_critic_value * (1 - dones[agent_idx])
                critic_loss = tf.keras.losses.MSE(target, critic_value)

            critic_grad = tape.gradient(critic_loss, agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(
                zip(critic_grad, agent.critic.trainable_variables))

            with tf.GradientTape() as tape:
                actions_pred = agent.actor(states[agent_idx])
                actions_pred_concat = actions.copy()
                actions_pred_concat[agent_idx] = actions_pred
                actions_pred_concat = tf.concat(actions_pred_concat, axis=1)

                actor_loss = -tf.math.reduce_mean(
                    agent.critic((states[agent_idx], actions_pred_concat)))

            actor_grad = tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(
                zip(actor_grad, agent.actor.trainable_variables))

            agent.update_network_parameters()


class MADDPGAgent:
    def __init__(self, agent_id, state_dim, n_actions, n_agents, alpha, beta, gamma, tau):
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau

        self.actor = ActorNetwork(n_actions, state_dim)
        self.critic = CriticNetwork(state_dim, n_actions * n_agents)
        self.target_actor = ActorNetwork(n_actions, state_dim)
        self.target_critic = CriticNetwork(state_dim, n_actions * n_agents)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=actions.shape, mean=0.0, stddev=0.1)
            actions = tf.clip_by_value(actions, 0, 1)

        # Return discrete action index
        return int(tf.argmax(actions[0]))