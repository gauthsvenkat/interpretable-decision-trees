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

    def expected_depth(self):
        """Returns the expected depth of a decision of each policy"""
        for name, policy, trace in zip(self.policy_names, self.policies, self.flattened_policy_traces):

            # convert obervations to numpy array
            observations = np.array([oo for oo, _, _ in trace])

            # returns a matrix of size (n_samples, n_nodes) with 1 if the sample traversed the node
            try:
                node_indicator = policy.tree.decision_path(observations).toarray()
            except AttributeError: #simple policies don't have any trees so we are improvising this
                node_indicator = np.ones((len(observations), 1))

            # get the list of node_ids that are traversed by the samples
            decision_paths = list(map(lambda x: np.where(x == 1)[0], node_indicator))

            # find the depth traversed by each sample
            depths = list(map(lambda x: len(x)-1, decision_paths))

            print('{} expected depth of decision {}'.format(name, np.mean(depths)))

    def feature_uniqueness(self):
        """Returns the ratio of the number of unique decision features used to the total number of features. Value close to 1 is better"""
        for name, policy, trace in zip(self.policy_names, self.policies, self.flattened_policy_traces):
            # convert obervations to numpy array
            observations = np.array([oo for oo, _, _ in trace])
            # returns a matrix of size (n_samples, n_nodes) with 1 if the sample traversed the node
            try:
                node_indicator = policy.tree.decision_path(observations).toarray()
            except AttributeError: #simple policies don't have any trees so we are improvising this
                node_indicator = np.ones((len(observations), 1))

            # get the list of node_ids that are traversed by the samples
            decision_paths = list(map(lambda x: np.where(x == 1)[0], node_indicator))

            #adjusted depth calculation. Prevents division by zero error.
            depths_mod = list(map(lambda x: max(len(x)-1.0,1), decision_paths))

            #list of features encountered while traversing the decision tree
            features = []

            #Captures repetitive use of the same features.
            feature_uniqueness_ratio = []

            for i in range(len(decision_paths)):
                features.append({}) #for each observation track the list of features encountered when making the decision
                for j in range(len(decision_paths[i])-1):
                    feature = policy.tree.tree_.feature[decision_paths[i][j]]
                    if(feature not in features[i].keys()): #maintain a dictionary of (feature number, occurrences) for each observation
                        features[i][feature] = 1
                    else:
                        features[i][feature] += 1

                #return the number of unique features used/ total number of features used for obtaining the action for an observation
                feature_uniqueness_ratio.append(len(features[i].keys())/depths_mod[i])

            print('{} expected uniqueness ratio {}'.format(name, np.mean(feature_uniqueness_ratio)))

    def node_counts(self):
        #simple metric. Returns the number of nodes in the tree
        for name,policy in zip(self.policy_names, self.policies):
            try:
                n_nodes = policy.tree.tree_.node_count
            except AttributeError: #no tree for simple policy
                n_nodes = 0
            print('{} decision nodes used: {}'.format(name, n_nodes))

    def tree_completeness_ratio(self):
        #Returns ratio of number of nodes to the max possible number of nodes. Smaller value is better
        for name,policy in zip(self.policy_names, self.policies):
            try:
                n_nodes = policy.tree.tree_.node_count
                max_depth = policy.tree.tree_.max_depth
                max_nodes = (2**(max_depth+1)) - 1
                completeness_ratio = n_nodes/max_nodes
            except AttributeError: #no tree for simple policy
                completeness_ratio = 1.0
            print("{} completeness ratio: {}".format(name,completeness_ratio))




    def evaluate(self):
        self.play_performance()
        self.fidelity()
        self.expected_depth()
        self.feature_uniqueness()
        self.node_counts()
        self.tree_completeness_ratio()
