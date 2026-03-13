# https://www.deep-ml.com/problems/37

'''
A correlation matrix is a table showing the correlation coefficients between variables. 
Each cell in the table shows the correlation between two variables, with values ranging from -1 to 1. 
These values indicate the strength and direction of the linear relationship between the variables.

corr(x,y) = cov(x,y) / (std(x) * std(y))
'''

import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X, dtype=float)

    if Y is None:
        # Standard correlation matrix
        Xc = X - X.mean(axis=0)
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)

    Y = np.asarray(Y, dtype=float)

    # Center
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # Sample covariance
    cov_xy = (Xc.T @ Yc) / (X.shape[0] - 1)

    std_x = np.sqrt(np.sum(Xc**2, axis=0) / (X.shape[0] - 1))
    std_y = np.sqrt(np.sum(Yc**2, axis=0) / (Y.shape[0] - 1))

    corr = cov_xy / np.outer(std_x, std_y)
    return corr


def main():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    print(X.shape)
    output = calculate_correlation_matrix(X)
    print(output)
main()
