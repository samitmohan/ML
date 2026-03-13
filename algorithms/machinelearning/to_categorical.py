# https://www.deep-ml.com/problems/34

'''
Write a Python function to perform one-hot encoding of nominal values. 
The function should take in a 1D numpy array x of integer values and an optional integer n_col representing the number of columns for the one-hot encoded array. 
If n_col is not provided, it should be automatically determined from the input array.

x = np.array([0, 1, 2, 1, 0])
    output = to_categorical(x)
    print(output)

# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]

Each element in the input array is transformed into a one-hot encoded vector, where the index corresponding to the value in the input array is set to 1, and all other indices are set to 0.
'''

import numpy as np

def to_categorical(x, n_col=None):
    num_classes = n_col if n_col is not None else len(set(x))
    # print(x.shape[0])
    encoding = np.zeros((x.shape[0], num_classes), dtype=int)
    # fill this encoding

    for i, val in enumerate(x):
        encoding[i][val] = 1
    return encoding

def main():
    print(to_categorical(np.array([0, 1, 2, 1, 0]), 4))

main()