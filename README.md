# ML-CSCI567
 Machine Learning models - **Python** + **NumPy** (No Additional Libraries)
  
# Supervised Learning
## 1. K Nearest Neighbour
Classified Medical record dataset to determine whether a person has Heart Disease or not.

**Functionalities -** Train, Get_K_Neighbours, Predict, Normalization_Scaler, MinMax_Scaler, Hyperparameter_Tuner  
**Distance Metrics used -** Euclidean, Minkowski, Inner Product, Gaussian Kernel, Cosine  
**Evaluation Metric -** F1 Score  
**Data Transformations -** Normalizing Feature Vector, Min-max Scaling on Feature Vector  
**Hyperparameter Tuning done on -** Distance metrics, Best K (No of classes), Best Data Transformation  

## 2. Decision Tree
Implemented **ID3 Algorithm** to classify discrete features from Car Dataset.

**Functionalities -** TreeNode_Split, Predict, Pruning  
**Measurement Metrics -** Information Gain, Entropy  
**Pruning Method -** Reduced Error Pruning (Post-pruning)  

## 3. Linear Regression
Predicted Wine Quality based on various features after **Regularization** to avoid Overfitting.  

**Functionalities -** Regression_without_Regularization, Regression_with_Regularization, Handle_Non-Invertible_Matrix, Tune_lambda,  Polynomial_Regression  
**Error Metric -** Mean Absolute Error (Mean Square Error)  
**Regularization method -** L2 Regularization   
**Mapping -** Polynomial Regression on data  

## 4. Binary Classification
Classified Synthetic and Two Moon data.  

**Loss Functions -** Perceptron Loss, Logistic Loss (Sigmoid)  

## 5. Multiclass Classification
Classified Toy dataset.

**Loss Function -** Multiclass Logistic Loss  (Softmax)  
**Optimization -** Gradient Descent, Stochastic Gradient Descent  

## 6. Neural Networks / Multi-Layer Perceptron
Classified **MNIST database of Handwritten digits** by computing **Derivates** for Forward and Backward passes.

**Functionalities -** Forward_learning, Error_Backpropagation  
**Mathematical Operations / Modules -** Linear_1, Relu / Tanh, Dropout, Linear_2, Softmax  
**Loss Function -** Cross-Entropy Loss  
**Optimization -** Gradient descent, Mini batch Stochastic Gradient Descent  
**Regularization Method -** Dropout  

# Unsupervised Learning
## K-Means
### Datasets ###
1. Toy Dataset - 2D (Generated from 4 Gaussian Distributions).  
2. Digits Dataset - From sklearn (8 x 8 per digit - 10 classes).  

**Functionalities -** Get_KMeans++-Centroids, Fit_Data_to_Centroids, Predict_Centroid_for_Data, Transform_Image.

### 1. Implement K-Means++ Clustering ###
Cluster centers were chosen based on high probability of them being apart.
### 2. Cluster Identification ###
Nearest cluster to the datapoints were identified.
### 3. Implement Classfication ###
Every datapoint belongs to the class of its cluster's centroid.
### 4. Image Compression ###
Performed Lossy Image Compression by replacing each cluster with its centroid.


# Probabilistic Learning
## 1. Hidden Markov Model
