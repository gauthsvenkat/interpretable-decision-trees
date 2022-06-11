from interpretableai import iai
import pydotplus
from sklearn import tree
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
    def __init__(self, max_depth, optimal_tree=False, cp=0):
        self.max_depth = max_depth
        self.optimal_tree = optimal_tree
        self.cp = cp if self.optimal_tree else None
        #doesn't matter if sklearn tree at this stage
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)


    def fit(self, obss, acts):

        #fit a normal decision tree classifier to the data
        self.tree.fit(obss, acts) 

        # if optimal tree, then fit an optimal tree classifier and perform model surgery on the sklearn fitted tree
        if self.optimal_tree:
            iclf = self._train_optimal_tree(obss, acts)
            iclf_state = self._extract_optimal_tree_state(iclf, obss, acts)
            self._perform_model_surgery(iclf_state)


    def _train_optimal_tree(self, obss, acts):
        iclf = iai.OptimalTreeClassifier(max_depth=self.max_depth, cp=self.cp)
        iclf.fit(obss, acts)
        return iclf


        # if optimal tree, then fit an optimal tree classifier and perform model surgery on the sklearn fitted tree
        if self.optimal_tree:
            iclf = self._train_optimal_tree(obss, acts)
            iclf_state = self._extract_optimal_tree_state(iclf, obss, acts)
            self._perform_model_surgery(iclf_state)


    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.fit(obss_train, acts_train)
        print('Train accuracy: {}'.format(self.accuracy(obss_train, acts_train)))
        print('Test accuracy: {}'.format(self.accuracy(obss_test, acts_test)))
        print('Number of nodes: {}'.format(self.tree.tree_.node_count))


    def predict(self, obss):
        vs = self.tree.predict(obss)
        return vs


    def _gini_impurity(self, counts_at_node):
        return 1 - np.sum(np.square(counts_at_node / counts_at_node.sum()))


    def _class_counts_at_node(self, labels_at_node, all_classes):

        counts_array = np.zeros(shape=(1, len(all_classes)))
        curr_class, counts = np.unique(labels_at_node, return_counts=True)

        for i in range(len(all_classes)):
            if all_classes[i] in curr_class:
                counts_array[0, i] = counts[curr_class == all_classes[i]]

        return counts_array


    def _optimal_tree_node_info(self, iclf, X_train, y_train, node_idx):
        if iclf.is_leaf(node_idx):
            left_child = right_child = -1
            feature = threshold = -2
        else:
            left_child = iclf.get_lower_child(node_idx) - 1 #-1 because 1-indexed
            right_child = iclf.get_upper_child(node_idx) - 1
            feature = int(iclf.get_split_feature(node_idx).replace('x', '')) - 1
            threshold = iclf.get_split_threshold(node_idx)
        #get all the unique classes, not just the ones in the node
        all_classes = np.unique(y_train)

        #get the indices and labels of the datapoints which pass through the current node
        data_indices_at_node = iclf.apply_nodes(X_train)[node_idx-1] #-1 because 1-indexed
        labels_at_node = y_train[data_indices_at_node]

        #get the counts of each class at the current node
        class_counts_at_node = self._class_counts_at_node(labels_at_node, all_classes)

        #get the impurity of the current node
        impurity = self._gini_impurity(class_counts_at_node)

        n_node_samples = len(labels_at_node)

        node_info = (left_child, right_child, feature, threshold, impurity, n_node_samples, n_node_samples)

        return node_info, class_counts_at_node


    def _extract_optimal_tree_state(self, iclf, X_train, y_train):

        nodes = []
        values = []

        for node_idx in range(1, iclf.get_num_nodes()+1):
            node, value = self._optimal_tree_node_info(iclf, X_train, y_train, node_idx)
            nodes.append(node)
            values.append(value)

        nodes = np.array(nodes, dtype=[('left_child', '<i8'),
                                       ('right_child', '<i8'),
                                       ('feature', '<i8'),
                                       ('threshold', '<f8'),
                                       ('impurity', '<f8'),
                                       ('n_node_samples', '<i8'),
                                       ('weighted_n_node_samples', '<f8')])
        values = np.array(values)

        iclf_state = {
            'max_depth': iclf.max_depth,
            'node_count': iclf.get_num_nodes(),
            'nodes': nodes,
            'values': values
        }

        return iclf_state


    def _perform_model_surgery(self, iclf_state):
        n_features = self.tree.n_features_
        n_classes = np.atleast_1d(self.tree.n_classes_)
        n_outputs = self.tree.n_outputs_

        #get a new Tree() object and set its state to the optimal tree's state
        donor_tree_object = tree._tree.Tree(n_features, n_classes, n_outputs)
        donor_tree_object.__setstate__(iclf_state)
        #transplant (hopefully) successful
        self.tree.tree_ = donor_tree_object


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
