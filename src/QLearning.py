# Q-learning algorithm adapted from https://github.com/guillaumefrd/q-learning-mountain-car.

import os
from pathlib import Path

import gym
import numpy as np
import pickle as pk


class EnvConfiguration:

    def __init__(self, n_episodes, num_bins, num_actions, obs_spaces, epsilon_decay_rate=None, min_eps=0.0):
        self.n_episodes = n_episodes
        self.num_bins = num_bins
        self.num_actions = num_actions
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_eps = min_eps

        # turn the continuous state space into a discrete space (with bins)
        # for the two observations: car position and car velocity
        if obs_spaces:
            self.discrete_states = [np.linspace(obs_space[0], obs_space[1], num=(self.num_bins + 1))[1:-1] for obs_space in obs_spaces]

            num_states = self.num_bins ** len(self.discrete_states)
            self.q_shape = (num_states, self.num_actions)

    def to_state(self, observation):
        # turn the observation features into a space represented by an integer
        states = [np.digitize(feature, self.discrete_states[i])
                    for i, feature in enumerate(observation)]
        states = [n * (self.num_bins ** i) for i, n in enumerate(states)]
        return sum(states)

    @staticmethod
    def get_env_config(env_name):
        e = gym.make(env_name)

        obs_space = e.observation_space
        n_actions = e.action_space.n
        e.close()
        obs_spaces = [(low, high) for low, high in zip(obs_space.high, obs_space.low)]

        if env_name == 'Acrobot-v1':
            n_buckets = 20
            n_episodes = 20000
            return EnvConfiguration(n_episodes, n_buckets, n_actions, obs_spaces)

        if env_name == 'MountainCar-v0':
            n_buckets = 10
            n_episodes = 75000
            epsilon_decay_rate = 5e-4
            return EnvConfiguration(n_episodes, n_buckets, n_actions, obs_spaces, epsilon_decay_rate)

        if env_name == 'CartPole-v1':
            n_buckets = 40
            n_episodes = 100000
            epsilon_decay_rate = 5e-4
            min_eps = 0.05
            return EnvConfiguration(n_episodes, n_buckets, n_actions, obs_spaces, epsilon_decay_rate, min_eps)


class QLearning:
    def __init__(self,
                 env_config,
                 lr_init=0.3,
                 lr_min=0.1,
                 lr_decay_rate=5e-4,
                 gamma=0.98,
                 epsilon=0.3,
                 ):

        self.env_config = env_config

        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_decay_rate = lr_decay_rate
        self.gamma = gamma
        self.epsilon = epsilon
        if env_config.epsilon_decay_rate:
            self.epsilon_decay_rate = env_config.epsilon_decay_rate
        else:
            self.epsilon_decay_rate = self.epsilon / (0.5 * env_config.n_episodes)
        self.state = None
        self.action = None

        self.q = np.zeros(shape=env_config.q_shape)

    def start_episode(self, observation):
        # apply decay on exploration
        self.epsilon = max(self.env_config.min_eps, self.epsilon * (1 - self.epsilon_decay_rate))
        # self.epsilon -= self.epsilon_decay_rate

        # apply decay on learning rate
        self.lr = max(self.lr_min, self.lr * (1 - self.lr_decay_rate))

        # return the first action of the episode
        self.state = self.env_config.to_state(observation)
        return np.argmax(self.q[self.state])

    def make_action(self, observation, reward=0):
        next_state = self.env_config.to_state(observation)

        if (1 - self.epsilon) <= np.random.uniform():
            # make a random action to explore
            next_action = np.random.randint(0, self.env_config.num_actions)
        else:
            # take the best action
            next_action = np.argmax(self.q[next_state])

        # update the Q-table
        self.q[self.state, self.action] += self.lr * \
                                           (reward + self.gamma * np.max(self.q[next_state, :]) -
                                            self.q[self.state, self.action])

        self.state = next_state
        self.action = next_action
        return next_action

    def predict(self, obss):
        return np.array([np.argmax(self.q[self.env_config.to_state(obs)]) for obs in obss])

    def predict_q(self, obss):
        return [self.q[self.env_config.to_state(obs)] for obs in obss]

    def train(self, env):
        rewards = []
        for episode_index in range(self.env_config.n_episodes):
            observation, done, total_reward = env.reset(), False, 0
            action = self.start_episode(observation)

            while not done:
                # make an action and get the new observations
                observation, reward, done, info = env.step(action)
                total_reward += reward

                # compute the next action
                action = self.make_action(observation, reward)
                # env.render()

            rewards.append(total_reward)

            if episode_index % 100 == 0:
                print("Episode {} average reward: {}; epsilon = {}, lr = {}".format(episode_index, sum(rewards) / len(rewards), self.epsilon, self.lr))
                rewards = []
        env.close()

    @staticmethod
    def get_oracle(env, specifier=""):
        path = Path('QLearning', env.unwrapped.spec.id, 'oracle{}.pk'.format(specifier))
        if path.exists():
            with open(path, 'rb') as f:
                return pk.load(f)

        oracle = QLearning(EnvConfiguration.get_env_config(env.unwrapped.spec.id))
        oracle.train(env)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, 'wb') as f:
            pk.dump(oracle, f)
        return oracle

