import numpy as np

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    weighted_entropy = 0  # H(S|A)
    np.seterr(divide='ignore', invalid='ignore')
    total_elements = sum(sum(x) for x in branches)
    for attribute_value in branches:
        weighted_entropy += (np.divide(calculate_entropy(attribute_value) * np.sum(attribute_value), total_elements, dtype=float))
    return S - weighted_entropy  # H(S) - H(S|A)


def calculate_entropy(attribute_value):
    attribute_label_probability = np.divide(attribute_value, sum(attribute_value), dtype=float)
    return -np.sum(np.nan_to_num(attribute_label_probability * np.log2(attribute_label_probability)))


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    prune_object = Prune(decisionTree, X_test, y_test)
    prune_object.prune_nodes(decisionTree.root_node)
    return

class Prune:
    def __init__(self, decisionTree, X_test, y_test):
        self.decisionTree = decisionTree
        self.X_test = X_test
        self.y_test = y_test
        # self.label_cnt = float(len(y_test))
        self.old_accuracy = self.calculate_accuracy()

    def calculate_accuracy(self):
        return len([x for x, value in enumerate(self.decisionTree.predict(self.X_test)) if value == self.y_test[x]])
        # return cnt / self.label_cnt

    def can_prune(self):
        accuracy = self.calculate_accuracy()
        if self.old_accuracy <= accuracy:
            self.old_accuracy = accuracy
            return True
        return False

    def prune_nodes(self, current_node):
        if current_node.children:
            for child in current_node.children:
                self.prune_nodes(child)

        if current_node.splittable:
            current_node.splittable = False
            if self.can_prune() is False:
                current_node.splittable = True

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass: ' + str(node.cls_max))
    print(indent + '}')
