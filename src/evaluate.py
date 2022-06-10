from pathlib import Path

import gym
import numpy as np
import math

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

        flat_obs = [obs for obs, _, _ in trace for trace in self.flattened_policy_traces]
        self.estimated_domains = list(zip(np.min(flat_obs, axis=0), np.min(flat_obs, axis=0)))

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

    def feature_importance_score(self):
        # It's better to have a few important features and not or barely use the others, users only need to keep the important ones in mind
        for name, policy in zip(self.policy_names, self.policies):
            # if the importance is concentrated over little features, they will have relatively high importance and thus low (1-importance)
            # this means the product of all (1-importance)s will be quite low
            try:
                fis = 1 - np.prod(1-policy.tree.tree_.compute_feature_importances())
            except AttributeError:
                fis = 0
            print("{} has importance score {}".format(name, fis))

    def insignificant_splits_ratio(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                max_sample = policy.tree.tree_.n_node_samples[0]
                non_leaves = policy.tree.tree_.children_left != -1
                useless = policy.tree.tree_.n_node_samples[non_leaves] < max_sample * 0.01  # splits on less than 1 percent of samples are annoying
            except AttributeError:
                useless = 0
                non_leaves = [1]
            print("{} has {}% significant splits".format(name, 100 * (1 - useless / np.sum(non_leaves))))

    def exact_feature_uniqueness(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                paths = [[0]]
                uniques = []
                while paths:
                    cur = paths.pop()
                    l = policy.tree.tree_.children_left[cur[-1]]
                    r = policy.tree.tree_.children_right[cur[-1]]
                    if (l == -1 == r):
                        uniques.append(np.unique(policy.tree.tree_.feature[cur]).size / cur.size)
                    else:
                        cur_ = cur.copy()
                        cur.append(r)
                        paths.append(cur)

                        cur_.append(l)
                        paths.append(cur_)
            except AttributeError:
                uniques = [1]
            print("{} had {}% unique features in each path".format(name, np.average(uniques) * 100))

    def unnecessary_splits(self):
        for name, policy in zip(self.policy_names, self.policies):
            non_leaves = policy.tree.tree_.children_left != -1
            try:
                print("{} needs {}% of nodes to model current behaviour".format(name, 100 * (1 - self._unnecessary_splits(policy, 0)[2] / np.sum(non_leaves))))
            except AttributeError:
                print("{} couldn't prune".format(name))

            
    def _unnecessary_splits(self, policy, node):
        l = policy.tree.tree_.children_left[node]
        r = policy.tree.tree_.children_right[node]
        if l == -1 == r:
            return True, policy.tree.tree_.value[node].argmax(), 0
        lu,lc,ln = self._unnecessary_splits(policy, l)
        ru,rc,rn = self._unnecessary_splits(policy, r)

        if not ru or not lu or lc != rc:
            return False, None, ln+rn

        return True, lc, ln+rn+1


    def same_feature_threshold_differences_in_path(self):
        for name, policy in zip(self.policy_names, self.policies):
            paths = [[0]]
            stds = []
            try:
                while paths:
                    cur = paths.pop()
                    l = policy.tree.tree_.children_left[cur[-1]]
                    r = policy.tree.tree_.children_right[cur[-1]]
                    if (l == -1 == r):
                        _features = policy.tree.tree_.feature[cur]
                        for f in np.unique(_features[_features >= 0]):
                            values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                            if values.size > 1:
                                values.sort()
                                domain = self.estimated_domains[f]
                                values = np.insert(np.insert(values, values.size, domain[1]), 0, domain[0])  # add bounds
                                diffs = np.abs(values[1:] - values[:-1])  # differences between neighbors
                                # the best case scenario is if all differences are domain/(num_thresholds + 1). Multiplying should thus give (domain/(num_thresholds + 1))^num_thresholds
                                # by seeing how far it is from this value we get an idea of how badly they're distributed. Note that an uneven distribution inherently causes the result to be lower
                                # this can be calculated by multiplying all (diff*(num_thresholds+1) / domain)
                                stds.append(np.prod(diffs * diffs.size / (domain[1] - domain[0])))
                    else:
                        cur_ = cur.copy()
                        cur.append(r)
                        paths.append(cur)

                        cur_.append(l)
                        paths.append(cur_)
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                print("{} has no repeating features".format(name))
            else:
                print("{} has an average threshold score of {} among repeating features in a path (max {}, min {})".format(name, np.average(stds), np.max(stds), np.min(stds)))


    def same_feature_threshold_differences_in_trace(self):
        for name, policy, trace in zip(self.policy_names, self.policies, self.flattened_policy_traces):
            try:
                for t in trace:
                    observations = np.array([oo for oo, _, _ in trace])

                    # returns a matrix of size (n_samples, n_nodes) with 1 if the sample traversed the node
                    try:
                        node_indicator = policy.tree.decision_path(observations).toarray()
                    except AttributeError: #simple policies don't have any trees so we are improvising this
                        node_indicator = np.ones((len(observations), 1))

                    # get the list of node_ids that are traversed by the samples
                    decision_paths = list(map(lambda x: np.where(x == 1)[0], node_indicator))

                    for d in decision_paths:
                        _features = policy.tree.tree_.feature[decision_paths]
                        for f in np.unique(_features[_features >= 0]):
                            values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                            if values.size > 1:
                                values.sort()
                                domain = self.estimated_domains[f]
                                values = np.insert(np.insert(values, values.size, domain[1]), 0, domain[0])  # add bounds
                                diffs = np.abs(values[1:] - values[:-1])  # differences between neighbors
                                # the best case scenario is if all differences are domain/(num_thresholds + 1). Multiplying should thus give (domain/(num_thresholds + 1))^num_thresholds
                                # by seeing how far it is from this value we get an idea of how badly they're distributed. Note that an uneven distribution inherently causes the result to be lower
                                # this can be calculated by multiplying all (diff*(num_thresholds+1) / domain)
                                stds.append(np.prod(diffs * diffs.size / (domain[1] - domain[0])))
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                print("{} has no repeating features".format(name))
            else:
                print("{} has an average threshold score of {} among repeating features in a path (max {}, min {})".format(name, np.average(stds), np.max(stds), np.min(stds)))


    def evaluate(self):
        self.play_performance()
        self.fidelity()
        self.expected_depth()
        self.feature_uniqueness()
        self.node_counts()
        self.tree_completeness_ratio()
        self.feature_importance_score()
        self.insignificant_splits_ratio()
        self.exact_feature_uniqueness()
        self.unnecessary_splits()
        self.same_feature_threshold_differences_in_trace()
        self.same_feature_threshold_differences_in_path()
