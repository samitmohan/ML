# https://www.deep-ml.com/problems/313?returnTo=paths
import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = x.astype(float)
    numerical_grad = np.zeros_like(x)

    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    # Relative error (standard ML definition)
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = numerator / (denominator + 1e-12)

    return numerical_grad, relative_error
