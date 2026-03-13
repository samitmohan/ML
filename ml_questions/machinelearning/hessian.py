# https://www.deep-ml.com/problems/218

from typing import Callable
import numpy as np

def compute_hessian(
    f: Callable[[list[float]], float],
    point: list[float],
    h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using
    central finite differences.
    """
    x = np.array(point, dtype=float)
    n = x.size
    H = np.zeros((n, n))

    f_x = f(x.tolist())

    for i in range(n):
        for j in range(n):
            if i == j:
                # Second derivative w.r.t. x_i
                x_forward = x.copy()
                x_backward = x.copy()
                x_forward[i] += h
                x_backward[i] -= h

                H[i, i] = (
                    f(x_forward.tolist())
                    - 2 * f_x
                    + f(x_backward.tolist())
                ) / (h ** 2)
            else:
                # Mixed partial derivative
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h

                H[i, j] = (
                    f(x_pp.tolist())
                    - f(x_pm.tolist())
                    - f(x_mp.tolist())
                    + f(x_mm.tolist())
                ) / (4 * h ** 2)

    return H.tolist()
