'''
Notes

Cross-entropy is a widely used loss function in machine learning, particularly in classification problems. 
It measures the performance of a model by comparing the predicted probability distribution with the actual distribution. 
A lower cross-entropy value indicates a better model performance.

y' = actual
y = predicted
mean of -(y' * log(y)  + (1 - y') * log(1 - y)) 

Cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. 
For multi-class classification tasks, we use the categorical cross-entropy loss.

y_c is binary indicator (0/1) if class label c is correct classication for sample
p_c is predicted prob that sample belongs to class c
C = number of classes
loss = -sum(from c=1 to C) yc * log(pc)
We usually take average across multiple samples in a batch(axis=1)

'''



import numpy as np
actual = np.array([1,0,1,0])
predicted = np.array([0.8, 0.2, 0.6, 0.4]) 
cross_entropy = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)).mean()
print(cross_entropy) # 0.36


# For Binary Classification (two classes 0 or 1)
def bin_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

y_true = np.array([1,0,1,0])
y_pred = np.array([0.8, 0.2, 0.6, 0.4]) 
bce = bin_cross_entropy(y_true, y_pred)
print(bce) # 0.36

# For Multiclass : cross-entropy loss function is calculated across all classes using the one-hot encoded target vectors and the predicted probability distributions.

def multiclass_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
    return loss

y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])

mce = multiclass_cross_entropy(y_true, y_pred)
print(f"Multiclass Cross-Entropy: {mce:.4f}")

# Mostly used in logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_cross_entropy(X, y, weights):
    z = np.dot(X, weights)
    y_pred = sigmoid(z) # [0.62245933 0.75026011 0.84553473]
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
    return loss

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
weights = np.array([0.1, 0.2])

loss = logistic_regression_cross_entropy(X, y, weights)
print(f"Logistic Regression Cross-Entropy: {loss:.4f}")

# Cross-entropy is widely used as the loss function in neural networks for classification tasks, particularly in the final layer (output layer) where the softmax activation function is applied.

# Optimising Cross Entropy Loss
def gradient_descent(X, y, weights, learning_rate, num_iterations):
    for _ in range(num_iterations):
        y_pred = np.dot(X, weights)
        loss = multiclass_cross_entropy(y, y_pred)
        gradients = np.dot(X.T, y_pred - y) / X.shape[0]
        weights -= learning_rate * gradients
    return weights

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
weights = np.array([0.1, 0.2])
learning_rate = 0.01
num_iterations = 1000

# optimized_weights = gradient_descent(X, y, weights, learning_rate, num_iterations)
# print(f"Optimized Weights: {optimized_weights}")


# Problem On DeepML

# https://www.deep-ml.com/problems/134

'''
Implement a function that computes the average cross-entropy loss for a batch of predictions in a multi-class classification task. 
Your function should take in a batch of predicted probabilities and one-hot encoded true labels, then return the average cross-entropy loss. 
Ensure that you handle numerical stability by clipping probabilities by epsilon
'''

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    loss = -np.sum(true_labels * np.log(predicted_probs), axis=1).mean()
    return loss

def main():
    predicted_probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
    true_labels = [[1, 0, 0], [0, 1, 0]]
    print(compute_cross_entropy_loss(predicted_probs, true_labels))

main()