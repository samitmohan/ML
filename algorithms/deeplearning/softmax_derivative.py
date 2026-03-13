# https://www.deep-ml.com/problems/219?returnTo=paths


import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    x = np.array(x, dtype=float)

    # Numerically stable softmax
    exp_x = np.exp(x - np.max(x))
    s = exp_x / np.sum(exp_x)

    # Jacobian: diag(s) - s s^T
    J = np.diag(s) - np.outer(s, s)

    return J.tolist()
