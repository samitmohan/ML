# https://www.deep-ml.com/problems/310?returnTo=paths

import numpy as np
from math import factorial

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor (Maclaurin) series approximation for common functions.
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series
    
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    result = 0.0

    if func_name == 'exp':
        for k in range(n_terms):
            result += x**k / factorial(k)

    elif func_name == 'sin':
        for k in range(n_terms):
            result += ((-1)**k) * x**(2*k + 1) / factorial(2*k + 1)

    elif func_name == 'cos':
        for k in range(n_terms):
            result += ((-1)**k) * x**(2*k) / factorial(2*k)

    else:
        raise ValueError("func_name must be 'exp', 'sin', or 'cos'")

    return round(result, 6)
