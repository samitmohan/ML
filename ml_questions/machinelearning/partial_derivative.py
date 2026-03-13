# https://www.deep-ml.com/problems/215?returnTo=paths
import numpy as np


def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.

    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate

    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    if func_name == 'poly2d':
        # x^2y + xy^2 so p(d/dx = 2xy + y^2) and p(d/dy = x^2 +2yx)
        x, y = point
        dfdx = 2 * x * y + y ** 2
        dfdy = x ** 2 + 2 * y * x 
        return (dfdx, dfdy)

    elif func_name == "exp_sum":
        # p(d/dx = e^(x+y)) = e^(x+y)
        x,y = point
        val = np.exp(x+y)
        return (val, val)

    elif func_name == "product_sin":
        # f(x,y) = pd wrt y = x*sin(y) = 1 + cos(y) * x and wrt x = sin(y)
        x,y=point
        dfdx = np.sin(y)
        dfdy = np.cos(y) * x
        return (dfdx, dfdy)
    
    elif func_name == "poly3d":
        # 'poly3d': f(x,y,z) = x²y + yz²
        x, y, z = point
        dfdx = 2*x*y
        dfdy = x**2 + z**2
        dfdz = 2*y*z
        return (dfdx, dfdy, dfdz)
    
    elif func_name == "sqaured_error":
        # (x-y)^2 (just like mse) = 2(x-y) wrt x and -2(x-y) wrt y
        x,y=point
        dfdx = 2*(x-y)
        dfdy = -2*(x-y)
        return (dfdx, dfdy)
    
    else:
        raise ValueError(f"Unknown function name: {func_name}")




        