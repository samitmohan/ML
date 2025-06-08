# https://www.deep-ml.com/problems/24

"""
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features.
The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input.
It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.
"""

import math
import numpy as np


def sigmoid(z: float) -> float:
    sigm = 1 / (1 + math.exp(-z))
    result = round(sigm, 4)
    return result


def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    features_np = np.array(features)
    weights_np = np.array(weights)
    labels_np = np.array(labels)
    predicted = np.dot(features_np, weights_np) + bias
    probabilities = np.array([sigmoid(val) for val in predicted])
    mse = np.mean((labels_np - probabilities) ** 2)
    rounded_probabilities = np.round(probabilities, 4)
    rounded_mse = np.round(mse, 4)
    return rounded_probabilities.tolist(), rounded_mse


def main():
    print(single_neuron_model(
            features=[[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
            labels=[0, 1, 0],
            weights=[0.7, -0.4],
            bias=-0.1,
        )
    )

main()
