# https://www.deep-ml.com/problems/39

'''
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. 
The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.
A = np.array([1, 2, 3])
print(log_softmax(A))

array([-2.4076, -1.4076, -0.4076])
'''
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    a = softmax(scores) 
    return np.log(a)

def main():
   A = np.array([1, 2, 3])
   print(log_softmax(A))
main()

'''
More
Directly applying the logarithm to the softmax function can lead to numerical instability, especially when dealing with large numbers. To prevent this, we use the log-softmax function, which incorporates a shift by subtracting the maximum value from the input vector:

The log-softmax function is particularly useful in machine learning for calculating probabilities in a stable manner, especially when used with cross-entropy loss functions.

# Subtract the maximum value for numerical stability
scores = scores - np.max(scores)
return np.round(scores - np.log(np.sum(np.exp(scores))), 4)
'''