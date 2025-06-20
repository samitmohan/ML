# https://www.deep-ml.com/problems/29

"""
Notes
Write a Python function to perform a random shuffle of the samples in two numpy arrays, X and y, while maintaining the corresponding order between them.
The function should have an optional seed parameter for reproducibility.

Example:
Input:

X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])
    y = np.array([1, 2, 3, 4])

Output:

(array([[5, 6],
                    [1, 2],
                    [7, 8],
                    [3, 4]]),
             array([3, 1, 4, 2]))

Reasoning:

The samples in X and y are shuffled randomly, maintaining the correspondence between the samples in both arrays.

Random shuffling of a dataset is a common preprocessing step in machine learning to ensure that the data is randomly distributed before training a model.
This helps to avoid any potential biases that may arise from the order in which data is presented to the model.
"""

import numpy as np


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    return X[idx], y[idx]


def main():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])

    print(shuffle_data(X, y, seed=True))


main()
