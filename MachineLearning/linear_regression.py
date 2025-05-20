# https://www.deep-ml.com/problems/14

import numpy as np
"""
Regression recap
Find me the weights (theta) that best stretch/scale the inputs X to match the output y

y = mx + c or theta1*x + theta2
in vector form it's y = X * theta where X is matrix and theta is the parameters that best scale X to match output Y.
Can't invert X since it's not square (guaranteed) so we multiply by X^T both sides and get -:
    theta = (X^T * X)^-1 * X^T * y

"""

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    x_tranpose = np.transpose(X)
    theta = np.linalg.inv(x_tranpose @ X) @ x_tranpose @ y
    return np.round(theta, 4).tolist()

def main():
	print(linear_regression_normal_equation(X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]))
main()