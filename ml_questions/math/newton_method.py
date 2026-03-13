# https://www.deep-ml.com/problems/221?returnTo=paths
'''
Docstring for ml.deepml.math.newton_method
Implement Newton's method for finding the minimum of a function. Given functions that compute the gradient and Hessian at any point, iteratively update the position using the Newton step until convergence. Newton's method uses second-order information (curvature) to converge faster than gradient descent, often finding the minimum of quadratic functions in a single step.


'''
from typing import Callable
import numpy as np

def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> list[float]:
    """
    Find the minimum of a function using Newton's method.
    """
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        grad = np.array(gradient_func(x.tolist()), dtype=float)

        # Convergence check
        if np.linalg.norm(grad) < tol:
            break

        hess = np.array(hessian_func(x.tolist()), dtype=float)

        # Newton step: solve H * delta = grad
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            # Hessian not invertible → stop safely
            break

        x = x - delta

    return x.tolist()
