import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import pickle as pk
import os


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)

        self.expected_depth_of_each_batch = []

    def fit(self, obss, acts):
        self.tree.fit(obss, acts)
        # Everytime the policy changes reset the expected depth list
        self.expected_depth_of_each_batch = []

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.fit(obss_train, acts_train)
        print('Train accuracy: {}'.format(self.accuracy(obss_train, acts_train)))
        print('Test accuracy: {}'.format(self.accuracy(obss_test, acts_test)))
        print('Number of nodes: {}'.format(self.tree.tree_.node_count))

    def predict(self, obss):

        # returns a matrix of size (n_samples, n_nodes) with 1 if the sample traversed the node
        node_indicator = self.tree.decision_path(obss).toarray() 
        # get the list of node_ids that are traversed by the samples
        decision_paths = list(map(lambda x: np.where(x == 1)[0], node_indicator))
        # find the depth traversed by each sample
        depths = list(map(lambda x: len(x)-1, decision_paths))
        # add the expected depth to the list
        self.expected_depth_of_each_batch.append(np.mean(depths))
        
        # return the list of leaf_node_ids by the samples
        leaf_nodes = list(map(lambda x: x[-1], decision_paths))
        # return the class labels of the leaf_node_ids
        vs = np.array(list(map(lambda x: np.argmax(self.tree.tree_.value[x]), leaf_nodes)))

        return vs
    
    def expected_depth(self):
        return np.mean(self.expected_depth_of_each_batch)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone

    def save_dt_policy(self, path):
        if not path.parent.exists():
            os.makedirs(str(path.parent), exist_ok=True)
        with open(str(path), 'wb') as f:
            pk.dump(self, f)

    def save_dt_policy_viz(self, path):
        if not path.parent.exists():
            os.makedirs(str(path.parent), exist_ok=True)

        dot_data = export_graphviz(self.tree, filled=True, rounded=True, impurity=False, class_names=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png(str(path))

    def accuracy(self, obss, acts):
        return np.mean(acts == self.predict(obss))

    @staticmethod
    def load_dt_policy(path):
        with open(path, 'rb') as f:
            return pk.load(f)

class SimpleMCDT:
    def predict(self, obss):
        return [0 if obs[1] < 0.0 else 2 for obs in obss]

    def expected_depth(self):
        return 0.0

class SimpleCartPoleDT:
    def predict(self, obss):
        return [0 if obs[3] < 0.0 else 1 for obs in obss]

    def expected_depth(self):
        return 0.0

class SimpleAcrobotDT:
    def predict(self, obss):
        return [0 if obs[4] <= -0.02 else 2 for obs in obss]

    def expected_depth(self):
        return 0.0
