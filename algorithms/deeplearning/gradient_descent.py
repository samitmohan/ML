# https://www.deep-ml.com/problems/47
import numpy as np

# do not shuffle data

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    for i in range(n_iterations):
        if method == 'batch':
            errors = X @ weights - y
            # derivative of error = 2 * error * derivative of error with respect to weights = 2 * error * X = 1/2n * 2 * error * X = errors * X / n
            gradients = 2 * X.T @ errors / len(y) 
            weights = weights - learning_rate * gradients 
        elif method == 'stochastic':
            for i in range(len(y)):
                xi= X[i:i+1]
                yi= y[i:i+1]
                error = xi @ weights - yi
                gradients = 2 * xi.T @ error
                weights = weights - learning_rate * gradients
        elif method == 'mini_batch':
            # Shuffle data
            for i in range(0, len(y), batch_size):
                xi = X[i:i+batch_size]
                yi = y[i:i+batch_size]
                errors = xi @ weights - yi
                gradients = 2 * xi.T @ errors / len(yi)
                weights = weights - learning_rate * gradients
        else:
            raise ValueError("Method must be 'batch', 'stochastic', or 'mini_batch'")
    return weights

def main():
    # Sample data
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])

    # Parameters
    learning_rate = 0.01
    n_iterations = 1000
    batch_size = 2

# Initialize weights
    weights = np.zeros(X.shape[1])
    final_weights_batch = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
    final_weights_stochastic = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
    final_weights_mini_batch = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')

    print("Final weights (Batch GD):", final_weights_batch)
    print("Final weights (Stochastic GD):", final_weights_stochastic)
    print("Final weights (Mini-batch GD):", final_weights_mini_batch)

main()