# https://www.deep-ml.com/problems/309?returnTo=paths

import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Multiply polynomials
    product = np.convolve(f_coeffs, g_coeffs)

    # Derivative of product
    if len(product) == 1:
        return [0.0]

    derivative = [i * product[i] for i in range(1, len(product))]
    return np.round(derivative, 4).tolist()

