# https://www.deep-ml.com/problems/31
import numpy as np
'''
Write a Python function to divide a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. The function should return two subsets of the dataset: one with samples that meet the condition and another with samples that do not.

X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8], 
                  [9, 10]])
    feature_i = 0
    threshold = 5

[array([[ 5,  6],
                    [ 7,  8],
                    [ 9, 10]]), 
             array([[1, 2],
                    [3, 4]])]
'''
def divide_on_feature(X, feature_i, threshold):
    return X[X[:, feature_i] >= threshold], X[X[:, feature_i] < threshold]

