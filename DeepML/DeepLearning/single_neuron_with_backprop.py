# https://www.deep-ml.com/problems/25


'''
Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. 
The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. 
The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
'''

import numpy as np

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []
    for _ in range(epochs):
        # forward pass
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # backward pass
        error = predictions - labels # expected - actual
        sigmoid_derivative = predictions * (1 - predictions)
        delta = error * sigmoid_derivative
        partial_derivative_error_wrt_weights = (2 / len(labels)) * np.dot(features.T, delta) 
        partial_derivative_error_wrt_bias = (2 / len(labels)) * np.mean(delta)

        # update
        weights -= learning_rate * partial_derivative_error_wrt_weights
        bias -=  learning_rate * partial_derivative_error_wrt_bias

        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)


    return updated_weights.tolist(), updated_bias, mse_values

def test_train_neuron():
    # Test case 1
    features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = np.array([1, 0, 0])
    initial_weights = np.array([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2
    expected_output = ([0.1035, -0.1426], -0.0056, [0.3033, 0.2947])
    assert train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs) == expected_output, "Test case 1 failed"
    
    # Test case 2
    features = np.array([[1, 2], [2, 3], [3, 1]])
    labels = np.array([1, 0, 1])
    initial_weights = np.array([0.5, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 3
    expected_output = ([0.4893, -0.2301], 0.001, [0.21, 0.2087, 0.2076])
    assert train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_train_neuron()
    print("All train_neuron tests passed.")
