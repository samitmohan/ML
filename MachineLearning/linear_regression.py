# https://www.deep-ml.com/problems/14
# https://www.deep-ml.com/problems/15

import numpy as np

"""
Regression recap
Find me the weights (theta) that best stretch/scale the inputs X to match the output y

y = mx + c or theta1*x + theta2
in vector form it's y = X * theta where X is matrix and theta is the parameters that best scale X to match output Y.
Can't invert X since it's not square (guaranteed) so we multiply by X^T both sides and get -:
    theta = (X^T * X)^-1 * X^T * y
    Limitation : Large matrix computations.

For larger values : gradient desc
notes on gradient descent -> calculates min cost function of parameters
Linear regression is for linear functions, gradientdesc works for any loss function.
The goal is to find the model parameters (slope m and intercept b) that minimize this cost function

Using library
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(learning_rate='constant', eta0=alpha, max_iter=iterations, fit_intercept=False)
model.fit(X, y)
θ_opt = np.hstack([model.intercept_, model.coef_])
"""


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    x_tranpose = np.transpose(X)
    theta = np.linalg.inv(x_tranpose @ X) @ x_tranpose @ y
    return np.round(theta, 4).tolist()


# https://www.youtube.com/watch?v=sDv4f4s2SB8
"""
Take derivative of loss function wrt all parameters (slope, intercept) -> basically taking the gradient of LF
Pick random val for parameters
S3: Plug parameter values into gradient (derivatives)
Calculate step size = slope * learning rate
Calculate new parameters = old_parameter - step_size
Repeat S3
"""


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    """
    # adjusts model parameters to find the best-fit line for given data.
    # y = X * theta, instead of mx + C
    # calculate cost function (MSE) = 1/n (sum(yi - ypred)^2) = 1/n (sum(yi - (mx_i + c)^2))
    # find gradient of cost function (partial derivative with resp to m and c)
    # D_m = (-2/n) * sum(X * (Y - y_pred))
    # D_c = (-2/n) * sum(Y - y_pred)
    # in vector form
    # update parameters: m = m - alpha (partial deriv wrt m), c = c - alpha (partial deriv wrt c)
    """
    m, n = X.shape
    theta = np.zeros(n)  # slope
    for _ in range(iterations):
        y_pred = X @ theta
        error = y_pred - y
        gradients = 1 / m * (X.T @ error)
        theta -= alpha * gradients
    return np.round(theta, 4).tolist()


def main():
    print(linear_regression_normal_equation(X=[[1, 1], [1, 2], [1, 3]], y=[1, 2, 3]))
    print(
        linear_regression_gradient_descent(
            X=np.array([[1, 1], [1, 2], [1, 3]]),
            y=np.array([1, 2, 3]),
            alpha=0.01,
            iterations=1000,
        )
    )


main()
