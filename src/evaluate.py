from pathlib import Path

import gym
import numpy as np
import os
from src.DTPolicy import DTPolicy
from src.QLearning import QLearning
from src.viper import get_rollouts_as_list_of_lists
from src.NormalDistAreaMatcher import getMatchDegree
from scipy.stats import norm

class Evaluate:

    def __init__(self, env, oracle, policies, n_rollouts=500, policy_names=None, experiment="", no_print=False, optimal = False):
        self.env = env
        self.policies = policies
        self.oracle = oracle
        self.n_rollouts = n_rollouts
        self.policy_names = policy_names if policy_names else ["policy{}".format(i) for i in range(len(policies))]
        self.no_print = no_print
        if(self.no_print == False):
            print('getting {} rollouts'.format(n_rollouts))
        self.policy_traces = [get_rollouts_as_list_of_lists(env, p, n_rollouts) for p in self.policies]
        self.oracle_trace = get_rollouts_as_list_of_lists(env, oracle, n_rollouts)
        self.experiment_paths = {} #stores the filepath of the experiments
        self.experiment = experiment
        env_name = env.unwrapped.spec.id
        if(self.experiment != ""):#you want to save the results
            if(optimal):
                pth = "experiments/"+env_name+"-optimal"+"/"
            else:
                pth = "experiments/"+env_name+"/"
            isExist = os.path.exists(pth)
            if(isExist == False):
                os.makedirs(pth)
            for policy in self.policy_names:
                self.experiment_paths[policy] = open(pth+policy+"_rollouts_"+str(self.n_rollouts)+"_"+self.experiment+".txt","w")



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
        if(self.no_print == False):
            print("Oracle average: {:.4f} with std: {:.4f}".format(self._oracle_average_reward(), self._oracle_std()))
        for avg, std, name, trace in zip(self._policy_average_rewards(), self._policy_stds(), self.policy_names, self.policy_traces):
            min_rollout = min([sum(rew for _, _, rew in rollout) for rollout in trace])
            diff = self._oracle_average_reward() - avg
            mean_oracle = self._oracle_average_reward()
            mean_policy = avg
            std_oracle = self._oracle_std()
            std_policy = std
            match_score = getMatchDegree(mean_oracle,mean_policy,std_oracle,std_policy)

            if(self.no_print == False):
                print("{} avg: {:.4f} with std {:.4f}. Difference = {:.4f} and min = {:.4f}. Match score = {:.4f}".format(name, avg, std, diff, min_rollout,match_score))
            if(self.experiment != ""):
                self.experiment_paths[name].write("Oracle average: {:.4f} with std: {:.4f}\n".format(self._oracle_average_reward(), self._oracle_std()))
                self.experiment_paths[name].write("{} avg: {:.4f} with std {:.4f}. Difference = {:.4f} and min = {:.4f}. Match score = {:.4f}\n".format(name, avg, std, diff, min_rollout,match_score))


    def fidelity(self):
        """Returns the chance of the oracle doing the same thing as the parent"""
        for name, policy, trace in zip(self.policy_names, self.policies, self.flattened_policy_traces):
            fid1 = sum([oa == policy.predict(np.array([oo]))[0] for oo, oa, _ in self.flattened_oracle_trace]) / len(self.flattened_oracle_trace)
            #fid2 = sum([oa == self.oracle.predict(np.array([oo]))[0] for oo, oa, _ in trace]) / len(trace)
            if(self.no_print == False):
                print('{} fidelity in oracle trace {:.4f}'.format(name, (fid1)))
            #print('{} fidelity in policy trace {:.4f}%'.format(name, (fid2 * 100)))
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} fidelity in oracle trace {:.4f}\n'.format(name, (fid1)))


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


            if(self.no_print == False):
                print('{} expected depth of decision {:.4f}'.format(name, np.mean(depths)))
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} expected depth of decision {:.4f}\n'.format(name, np.mean(depths)))

    def expected_depth_score(self):
        """Returns the expected depth for taking an action divided by the max depth of the tree"""
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

            max_depth = policy.tree.tree_.max_depth
            if(self.no_print == False):
                print('{} expected depth score {:.4f}'.format(name, 1 - np.mean(depths)/max_depth)) #because a large ratio is bad. So we do 1 -
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} expected depth score {:.4f}\n'.format(name, 1 - np.mean(depths)/max_depth))



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

                """If the counts for two nodes are 10000 and 2, then effectively its as if there is only 1 node being used.
                The below function gets this effective node usage count. Earlier method using .keys() would have resulted
                in 2 effective nodes. We need something in between 1 an 2"""
                equi_nodes = self.getEffectiveFeaturesUsed(features[i].values())
                feature_uniqueness_ratio.append(equi_nodes/depths_mod[i])

            if(self.no_print == False):
                print('{} expected uniqueness ratio {:.4f}'.format(name, np.mean(feature_uniqueness_ratio)))
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} expected uniqueness ratio {:.4f}\n'.format(name, np.mean(feature_uniqueness_ratio)))

    def getEffectiveFeaturesUsed(self, usage_counts):
        """Example: Assume there are two nodes with occurrence counts 10000 and 2. 1/np.mean() results in 1/5001. Now arr becomes [10000,2] * 1/5001
         =  [1.99,3.99 x 10^-4]. Effectively only the first one should be counted. So, we threshold values to 1 then sum to get the effective count"""
        if len(usage_counts) == 0:
             return 0
        arr = np.array(list(usage_counts))
        arr = arr * (1.0/np.mean(arr)) #to see which values exceed the average.
        arr = np.clip(arr, 0, 1) #All nodes with values greater than the mean are counted as effective. So threshold at 1.
        return np.sum(arr)

    def node_counts(self):
        #simple metric. Returns the number of nodes in the tree
        for name,policy in zip(self.policy_names, self.policies):
            try:
                n_nodes = policy.tree.tree_.node_count
            except AttributeError: #no tree for simple policy
                n_nodes = 0
            if(self.no_print == False):
                print('{} decision nodes used: {}'.format(name, n_nodes))
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} decision nodes used: {}\n'.format(name, n_nodes))

    def node_counts_score(self):
        for name,policy in zip(self.policy_names, self.policies):
            try:
                n_nodes = policy.tree.tree_.node_count
            except AttributeError: #no tree for simple policy
                n_nodes = 0
            if(self.no_print == False):
                print('{} node count score: {:.4f}'.format(name, 1/(1+n_nodes)))
            if(self.experiment != ""):
                self.experiment_paths[name].write('{} node count score: {:.4f}\n'.format(name, 1/(1+n_nodes))) #normalizes node count score between zero and one. 1 is good


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
            if(self.no_print == False):
                print("{} completeness ratio: {:.4f}".format(name,completeness_ratio))
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} completeness ratio: {:.4f}\n".format(name,completeness_ratio))

    def tree_completeness_ratio_score(self):
        #Returns ratio of number of nodes to the max possible number of nodes. Smaller value is better
        for name,policy in zip(self.policy_names, self.policies):
            try:
                n_nodes = policy.tree.tree_.node_count
                max_depth = policy.tree.tree_.max_depth
                max_nodes = (2**(max_depth+1)) - 1
                completeness_ratio = n_nodes/max_nodes
            except AttributeError: #no tree for simple policy
                completeness_ratio = 1.0
            if(self.no_print == False):
                print("{} completeness ratio score: {:.4f}".format(name,1-completeness_ratio)) #this behaves similar to expected depth so 1 -
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} completeness ratio score: {:.4f}\n".format(name,1-completeness_ratio))


    def feature_importance_score(self):
        # It's better to have a few important features and not or barely use the others, users only need to keep the important ones in mind
        for name, policy in zip(self.policy_names, self.policies):
            # if there are little important features, 1-(importance) will have mostly high values so their product will be high
            try:
                fis = np.prod(1-policy.tree.tree_.compute_feature_importances())
            except AttributeError:
                fis = 0
            if(self.no_print == False):
                print("{} has importance score {:.4f}".format(name, fis))
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} has importance score {:.4f}\n".format(name, fis))

    def insignificant_leaves(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                max_sample = policy.tree.tree_.n_node_samples[0]
                useless = policy.tree.tree_.n_node_samples < max_sample * 0.01
                non_leaves = np.concatenate((policy.tree.tree_.children_left != -1, policy.tree.tree_.children_right != -1))
            except AttributeError:
                useless = 0
                non_leaves = [1]
            if(self.no_print == False):
                print("{} has {:.4f} insignificant splits".format(name, np.sum(useless) / np.sum(non_leaves)))
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} has {:.4f} insignificant splits\n".format(name, np.sum(useless) / np.sum(non_leaves)))

    def insignificant_leaves_score(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                max_sample = policy.tree.tree_.n_node_samples[0]
                useless = policy.tree.tree_.n_node_samples < max_sample * 0.01
                non_leaves = np.concatenate((policy.tree.tree_.children_left != -1, policy.tree.tree_.children_right != -1))
            except AttributeError:
                useless = 0
                non_leaves = [1]
            if(self.no_print == False):
                print("{} has {:.4f} insignificant leaves score".format(name, 1 - (np.sum(useless) / np.sum(non_leaves)))) #1 - coz more insignificant leaves is bad
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} has {:.4f} insignificant leaves score\n".format(name, 1 - (np.sum(useless) / np.sum(non_leaves))))

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
                        used_features = [policy.tree.tree_.feature[i] for i in cur]
                        features = {}
                        for i in cur:
                            used_feat = policy.tree.tree_.feature[i]
                            if(used_feat >= 0):#don't allow negative features because they are leaves
                                if(used_feat not in features.keys()):
                                    features[used_feat] = 1
                                else:
                                    features[used_feat] += 1


                        equi_nodes = self.getEffectiveFeaturesUsed(features.values())
                        uniques.append(equi_nodes/len(used_features))
                    else:
                        cur_ = cur.copy()
                        cur.append(r)
                        paths.append(cur)
                        cur_.append(l)
                        paths.append(cur_)
            except AttributeError:
                uniques = [1]
            if(self.no_print == False):
                print("{} had {:.4f} unique features in each path".format(name, np.average(uniques)))
            if(self.experiment != ""):
                self.experiment_paths[name].write("{} had {:.4f} unique features in each path\n".format(name, np.average(uniques)))

    def unnecessary_splits(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                if(self.no_print == False):
                    print("{} could prune {:.4f} of nodes without reducing performance".format(name, self._unnecessary_splits(policy, 0)[2] / policy.tree.tree_.node_count))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} could prune {:.4f} of nodes without reducing performance\n".format(name, self._unnecessary_splits(policy, 0)[2] / policy.tree.tree_.node_count))
            except AttributeError:
                if(self.no_print == False):
                    print("{} couldn't prune".format(name))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} couldn't prune\n".format(name))

    def unnecessary_splits_score(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                if(self.no_print == False):
                    print("{} unnecessary splits score: {:.4f}".format(name, 1 - (self._unnecessary_splits(policy, 0)[2] / policy.tree.tree_.node_count)))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} unnecessary splits score: {:.4f}\n".format(name, 1- (self._unnecessary_splits(policy, 0)[2] / policy.tree.tree_.node_count)))
            except AttributeError:
                if(self.no_print == False):
                    print("{} couldn't prune".format(name))
                if(self.experiment != ""):
                    pass #leave it


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

    def same_feature_value_differences(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                features = policy.tree.tree_.feature[policy.tree.tree_.feature >= 0]
                stds = []
                scores = []
                for f in np.unique(features):
                    values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                    if values.size > 1:
                        stds.append(np.std(values))
                        scores.append(np.std(values)/(np.max(values) - np.min(values))) #additional domain info use
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                if(self.no_print == False):
                    print("{} has no repeating features".format(name))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} has no repeating features\n".format(name))
            else:
                if(self.no_print == False):
                    print("{} path feature value difference score {:.4f}".format(name, np.average(scores)))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} path feature value difference score {:.4f}".format(name, np.average(scores)))


    def same_feature_value_differences_score(self):
        for name, policy in zip(self.policy_names, self.policies):
            try:
                features = policy.tree.tree_.feature[policy.tree.tree_.feature >= 0]
                stds = []
                scores = []
                for f in np.unique(features):
                    values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                    if values.size > 1:
                        stds.append(np.std(values))
                        scores.append(np.std(values)/(np.max(values) - np.min(values))) #additional domain info use
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                if(self.no_print == False):
                    print("{} has no repeating features".format(name))
                if(self.experiment != ""):
                    pass
            else:
                if(self.no_print == False):
                    print("{} path feature value difference score {:.4f}".format(name, np.average(scores)))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} path feature value difference score {:.4f}".format(name, np.average(scores)))

    def same_value_differences_in_path(self):
        for name, policy in zip(self.policy_names, self.policies):
            paths = [[0]]
            stds = []
            scores = []
            try:
                while paths:
                    cur = paths.pop()
                    l = policy.tree.tree_.children_left[cur[-1]]
                    r = policy.tree.tree_.children_right[cur[-1]]
                    if (l == -1 == r):
                        _features = policy.tree.tree_.feature[cur]
                        features = policy.tree.tree_.feature[policy.tree.tree_.feature >= 0]
                        for f in np.unique(features):
                            values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                            if values.size > 1:
                                stds.append(np.std(values))
                                scores.append(np.std(values)/(np.max(values) - np.min(values))) #additional domain info use
                    else:
                        cur_ = cur.copy()
                        cur.append(r)
                        paths.append(cur)

                        cur_.append(l)
                        paths.append(cur_)
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                if(self.no_print == False):
                    print("{} has no repeating features".format(name))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} has no repeating features\n".format(name))
            else:
                if(self.no_print == False):
                    print("{} path feature value difference score {:.4f}".format(name, np.average(scores)))
#                     print("{} path feature value difference score {:.4f}".format(name, np.average(stds)/np.max(stds)))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} path feature value difference score {:.4f}".format(name, np.average(scores)))
#                     self.experiment_paths[name].write("{} path feature value difference score {:.4f}".format(name, np.average(stds)/np.max(stds)))



    def same_value_differences_in_path_score(self):
        for name, policy in zip(self.policy_names, self.policies):
            paths = [[0]]
            stds = []
            scores = []
            try:
                while paths:
                    cur = paths.pop()
                    l = policy.tree.tree_.children_left[cur[-1]]
                    r = policy.tree.tree_.children_right[cur[-1]]
                    if (l == -1 == r):
                        _features = policy.tree.tree_.feature[cur]
                        features = policy.tree.tree_.feature[policy.tree.tree_.feature >= 0]
                        for f in np.unique(features):
                            values = policy.tree.tree_.threshold[policy.tree.tree_.feature == f]
                            if values.size > 1:
                                stds.append(np.std(values))
                                scores.append(np.std(values)/(np.max(values) - np.min(values))) #additional domain info use
                    else:
                        cur_ = cur.copy()
                        cur.append(r)
                        paths.append(cur)

                        cur_.append(l)
                        paths.append(cur_)
            except AttributeError:
                stds = [0]
            if len(stds) == 0:
                if(self.no_print == False):
                    print("{} has no repeating features".format(name))
                if(self.experiment != ""):
                    pass
            else:
                if(self.no_print == False):
#                     print("{} path feature value difference score {:.4f}".format(name, np.average(stds)/np.max(stds)))
                    print("{} path feature value difference score {:.4f}".format(name, np.average(scores)))
                if(self.experiment != ""):
                    self.experiment_paths[name].write("{} path feature value difference score {:.4f}".format(name, np.average(scores)))



    def evaluate(self,):
        self.play_performance()
        self.fidelity()
        self.expected_depth()
        self.expected_depth_score()
        self.feature_uniqueness()
        self.node_counts()
        self.node_counts_score()
        self.tree_completeness_ratio()
        self.tree_completeness_ratio_score()
        self.feature_importance_score()
        self.insignificant_leaves()
        self.insignificant_leaves_score()
        self.exact_feature_uniqueness()
        self.unnecessary_splits()
        self.unnecessary_splits_score()
        self.same_feature_value_differences()
        self.same_feature_value_differences_score()
        self.same_value_differences_in_path()
        self.same_value_differences_in_path_score()
        for file in self.experiment_paths.values():
            file.close()
