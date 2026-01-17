# https://www.deep-ml.com/problems/116?returnTo=paths
def poly_term_derivative(c: float, x: float, n: float) -> float:
    derivative = c * n * x**(n-1)
    return derivative
