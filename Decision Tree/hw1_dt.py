import numpy as np
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(list(feature))
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = np.array(features)
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        branch = []
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            branch.append(self.labels.count(label))
            if branch[-1] > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splittable is false when all features belongs to one class
        self.entropy = Util.calculate_entropy(branch)
        if len(np.unique(labels)) < 2 or len(self.features[0]) == 0:
            self.splittable = False
        else:
            self.splittable = True
        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        self.best_info_gain = float('-inf')
        self.best_attribute_values = []

        for attribute_index in range(len(self.features[0])):
            branch = dict()
            for attribute_value in sorted(set(self.features[:, attribute_index])):
                branch[attribute_value] = [0] * self.num_cls
            label_map = sorted(set(self.labels))
            for label_index, label in enumerate(self.labels):
                branch[self.features[label_index, attribute_index]][label_map.index(label)] \
                 = branch.get(self.features[label_index, attribute_index], 0)[label_map.index(label)] + 1
            current_info_gain = Util.Information_Gain(self.entropy, list(branch.values()))

            if (current_info_gain != 0) and ((current_info_gain > self.best_info_gain) or
                    (current_info_gain == self.best_info_gain and len(branch) > self.feature_uniq_split)):
                self.best_info_gain = current_info_gain
                self.dim_split = attribute_index
                self.feature_uniq_split = len(branch)
                self.best_attribute_values = list(branch.keys())

        if self.best_info_gain != float('-inf'):
            # split the best attribute and create children
            child_feature_array = np.column_stack((self.features[:, :self.dim_split], self.features[:, self.dim_split+1:]))
            for attribute_value in self.best_attribute_values:
                subset_of_indices = np.where(self.features[:, self.dim_split] == attribute_value)[0]
                child_labels = np.array(self.labels)[subset_of_indices].tolist()
                self.children.append(TreeNode(child_feature_array[subset_of_indices].tolist(), child_labels,\
                    len(set(child_labels))))
                if self.children[-1].splittable:
                    self.children[-1].split()
        else:
            self.splittable = False
        return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        # for attribute in
        if self.splittable and self.dim_split is z:
            attribute_value = feature.pop(self.dim_split)
            return self.children[self.best_attribute_values.index(attribute_value)].predict(feature)
        else:
            return self.cls_max
