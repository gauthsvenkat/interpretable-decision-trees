from interpretableai import iai

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
    def __init__(self, max_depth, optimal_tree=True):
        self.max_depth = max_depth
        self.optimal_tree = optimal_tree
        if self.optimal_tree:
            self.tree = iai.OptimalTreeClassifier(max_depth=self.max_depth)
        else :
            self.tree = DecisionTreeClassifier(max_depth=self.max_depth)

    def fit(self, obss, acts):
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)

        if self.optimal_tree:
            self.tree = iai.OptimalTreeClassifier(max_depth=self.max_depth)
        else :
            self.tree = DecisionTreeClassifier(max_depth=self.max_depth)

        self.fit(obss_train, acts_train)
        print('Train accuracy: {}'.format(self.accuracy(obss_train, acts_train)))
        print('Test accuracy: {}'.format(self.accuracy(obss_test, acts_test)))
        print('Number of nodes: {}'.format(self.tree.tree_.node_count))

    def predict(self, obss):
        vs = self.tree.predict(obss)
        return vs

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

class SimpleCartPoleDT:
    def predict(self, obss):
        return [0 if obs[3] < 0.0 else 1 for obs in obss]

class SimpleAcrobotDT:
    def predict(self, obss):
        return [0 if obs[4] <= -0.02 else 2 for obs in obss]
