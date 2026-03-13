# Vector to Diagonal matrix
# https://www.deep-ml.com/problems/35

import numpy as np

'''
Write a Python function to convert a 1D numpy array into a diagonal matrix. 
The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.
'''

def make_diagonal(x):
    n = len(x)
    diag_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diag_matrix[i][i] = x[i]
    return np.array(diag_matrix)

def main():
    x = np.array([1, 2, 3, 4, 5])
    print(make_diagonal(x))

main()