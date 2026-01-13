# https://www.deep-ml.com/problems/109?from=Attention%20Is%20All%20You%20Need

# mean and variance of the activations are calculated for each layer separately, and then the activations are scaled and shifted to have a standard normal distribution (mean of 0 and variance of 1).

# y_i = gamma * (x_i - mean) / std_dev + error + beta where gamma and beta are learnable parameters 

'''
Implement a function to perform Layer Normalization on an input tensor. 
Given a 3D array representing batch_size, sequence length, and feature dimensions, 
- normalize the data across the feature dimension for each sequence, then apply scaling and shifting parameters.


'''


import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # since shapes are (b, sl, features) we want mean and std_dev over features (hence axis=-1)
    mean = np.mean(X, axis=-1, keepdims=True)
    std_dev = np.std(X, axis=-1, keepdims=True)
    X_norm = (X - mean) / (std_dev + epsilon)

    output = gamma * X_norm + beta
    return output

def main():
    np.random.seed(42)
    X = np.random.randn(2, 2, 3)
    gamma = np.ones(3).reshape(1, 1, -1)
    beta = np.zeros(3).reshape(1, 1, -1)
    print(layer_normalization(X, gamma, beta))

main()

