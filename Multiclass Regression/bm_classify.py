import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    y[y == 0] = -1

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        for iter in range(max_iterations):
            yXw = np.multiply((np.matmul(X, w) + b), y)
            avg_grad_w = np.matmul(np.transpose(X), y * np.where(yXw <= 0, 1, 0)) / N
            avg_grad_b = np.sum(y[np.where(yXw <= 0)]) / N
            w = w + step_size * avg_grad_w
            b = b + step_size * avg_grad_b

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        for iter in range(max_iterations):
            sig_yXw = sigmoid(-np.multiply((np.matmul(X, w) + b), y))
            avg_grad_w = np.matmul(np.transpose(X), np.multiply(y, sig_yXw)) / N
            avg_grad_b = np.matmul(np.transpose(y), sig_yXw) / N
            w = w + step_size * avg_grad_w
            b = b + step_size * avg_grad_b

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """
    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    ############################################
    np.seterr(over='ignore')
    return np.divide(1, 1 + np.exp(-z))


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        ############################################
        preds = np.where((np.matmul(X, w) + b) > 0, 1, 0)

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        ############################################
        preds = np.where(sigmoid(np.matmul(X, w) + b) > 0.5, 1, 0)

    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        X = np.concatenate((X, np.ones(N).reshape(N, 1)), axis=1)
        W = np.concatenate((w, b.reshape(C, 1)), axis=1)
        ############################################
        for iter in range(max_iterations):
            n = np.random.choice(N)
            x = X[n].reshape(D + 1, 1)
            softmax_arr = softmax(np.matmul(W, x))
            softmax_arr[y[n]] -= 1
            W = W - step_size * np.matmul(softmax_arr, np.transpose(x))
        b = W[:, -1]
        w = W[:, :-1]

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        X = np.concatenate((X, np.ones(N).reshape(N, 1)), axis=1)
        W = np.concatenate((w, b.reshape(C, 1)), axis=1)
        ############################################
        y_class = np.transpose(np.eye(C)[y])
        for iter in range(max_iterations):
            softmax_arr = softmax(np.matmul(W, np.transpose(X)))
            W = W - (step_size * np.matmul(np.subtract(softmax_arr, y_class), X)) / N
        b = W[:, -1]
        w = W[:, :-1]

    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    print (w,b)
    return w, b


def softmax(z):
    ez = np.exp(z - np.max(z))
    return np.divide(ez, np.sum(ez, axis=0))


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    C, = b.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    X = np.concatenate((np.ones(N).reshape(N, 1), X), axis=1)
    W = np.concatenate((b.reshape(C, 1), w), axis=1)
    ############################################
    preds = np.argmax(softmax(np.matmul(W, np.transpose(X))), axis=0)
    assert preds.shape == (N,)
    return preds




