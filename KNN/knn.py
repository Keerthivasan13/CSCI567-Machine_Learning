import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features = [[]]
        self.labels = []

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels
        return

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        predicted_labels = []
        for test_instance in features:
            predicted_labels.append(Counter(self.get_k_neighbors(test_instance)).most_common()[0][0])

        return predicted_labels
        raise NotImplementedError

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        distances = []
        for x in range(len(self.features)):
            distances.append([self.distance_function(self.features[x], point), self.labels[x]])
        return (x[1] for x in sorted(distances, key=lambda x: x[0])[0:self.k])


if __name__ == '__main__':
    print(np.__version__)

    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.float_power(np.sum(np.float_power((np.asarray(point1) - np.asarray(point2)), 2)), 0.5)
        raise NotImplementedError


    def f1_score(real_labels, predicted_labels):
        """
        Information on F1 score - https://en.wikipedia.org/wiki/F1_score
        :param real_labels: List[int]
        :param predicted_labels: List[int]
        :return: float
        """
        assert len(real_labels) == len(predicted_labels)
        real = np.array(real_labels)
        predict = np.array(predicted_labels)

        tp = np.sum(np.logical_and(real, predict))
        # tn = len(real_labels) - sum(real | predict)
        fp = np.sum(np.logical_and(np.logical_xor(real, predict), predict))
        fn = np.sum(np.logical_and(np.logical_xor(real, predict), real))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (2 * precision * recall / (precision + recall))
"""
best_f1_score = float('-inf')
for k in range(1, 30, 2):
    model = KNN(k, euclidean_distance)
    model.train([[1, 2], [3, 4], [3, 4]], [0, 1, 1])
    predicted_labels = []
    for test_instance in [[3, 4]]:
        predicted_labels.append(model.predict(test_instance))
    score = f1_score([1], predicted_labels)
    print score
"""