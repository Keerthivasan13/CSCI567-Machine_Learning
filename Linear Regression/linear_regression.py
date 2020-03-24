"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    return np.mean(np.abs(np.subtract(np.matmul(X, w), y)))


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    matrix = np.matmul(np.transpose(X), X)
    while np.min(np.absolute(np.linalg.eigvals(matrix))) < (10 ** (-5)): # Its Non-invertible
        matrix = np.add(matrix, 0.1 * np.identity(np.size(matrix, 1)))
    return np.matmul(np.matmul(np.linalg.inv(matrix), np.transpose(X)), y)

###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    return np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(X), X), (lambd * np.identity(np.size(X, 1))))), np.transpose(X)), y)


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    bestlambda = None
    least_error = float('Inf')
    for lambd in range(-19, 20):
        w = regularized_linear_regression(Xtrain, ytrain, 10**lambd)
        error = mean_absolute_error(w, Xval, yval)
        if error < least_error:
            least_error = error
            bestlambda = lambd
    return 10**bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    res = X
    for expo in range(2, power+1):
        res = np.concatenate((res, np.power(X, expo)), axis=1)
    return res
