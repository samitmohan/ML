# https://www.deep-ml.com/problems/202?returnTo=paths

import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """
    x = np.array(x, dtype=float)
    fx = np.array(f(x), dtype=float)
    m = fx.size # op dimension
    n = x.size # input dim
    J = np.zeros((m, n))
    for j in range(n):
        xstep = x.copy()
        xstep[j] += h  # stepsize
        fstep = np.array(f(xstep), dtype=float)
        #pd
        J[:, j] = (fstep-fx) / h 
    return J.tolist()

