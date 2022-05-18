from pathlib import Path

import gym
import numpy as np

from src.DTPolicy import DTPolicy
from src.QLearning import QLearning
from src.viper import get_rollouts_as_list_of_lists


class Evaluate:

    def __init__(self, env, oracle, policies, n_rollouts=500, policy_names=None):
        self.env = env
        self.policies = policies
        self.oracle = oracle
        self.n_rollouts = n_rollouts
        self.policy_names = policy_names if policy_names else ["policy{}".format(i) for i in range(len(policies))]
        print('getting {} rollouts'.format(n_rollouts))
        self.policy_traces = [get_rollouts_as_list_of_lists(env, p, n_rollouts) for p in self.policies]
        self.oracle_trace = get_rollouts_as_list_of_lists(env, oracle, n_rollouts)

    def _policy_average_rewards(self):
        avgs = []
        for trace in self.flattened_policy_traces:
            avg = sum([rew for _, _, rew in trace]) / self.n_rollouts
            avgs.append(avg)
        return avgs

    def _oracle_average_reward(self):
        avg = sum([rew for _, _, rew in self.flattened_oracle_trace]) / self.n_rollouts
        return avg

    def _oracle_std(self):
        return np.std([sum(rew for _, _, rew in trace) for trace in self.oracle_trace])

    def _policy_stds(self):
        stds = []
        for trace in self.policy_traces:
            stds.append(np.std([sum(rew for _, _, rew in trace) for trace in trace]))
        return stds

    @property
    def flattened_oracle_trace(self):
        return [v for trace in self.oracle_trace for v in trace]

    @property
    def flattened_policy_traces(self):
        traces = []
        for full_trace in self.policy_traces:
            traces.append([v for trace in full_trace for v in trace])
        return traces

    def play_performance(self):
        """Returns difference in average reward for the two policies"""
        print("Oracle average: {} with std: {}".format(self._oracle_average_reward(), self._oracle_std()))
        for avg, std, name, trace in zip(self._policy_average_rewards(), self._policy_stds(), self.policy_names, self.policy_traces):
            min_rollout = min([sum(rew for _, _, rew in rollout) for rollout in trace])
            diff = self._oracle_average_reward() - avg
            print("{} avg: {} with std {}. Difference = {} and min = {}".format(name, avg, std, diff, min_rollout))

    def fidelity(self):
        """Returns the chance of the oracle doing the same thing as the parent"""
        for name, policy, trace in zip(self.policy_names, self.policies, self.flattened_policy_traces):
            fid1 = sum([oa == policy.predict(np.array([oo]))[0] for oo, oa, _ in self.flattened_oracle_trace]) / len(self.flattened_oracle_trace)
            fid2 = sum([oa == self.oracle.predict(np.array([oo]))[0] for oo, oa, _ in trace]) / len(trace)

            print('{} fidelity in oracle trace {}%'.format(name, (fid1 * 100)))
            print('{} fidelity in policy trace {}%'.format(name, (fid2 * 100)))

    def stability(self, env, policy):
        pass

    def evaluate(self):
        self.play_performance()
        self.fidelity()
