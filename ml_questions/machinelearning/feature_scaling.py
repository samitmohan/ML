# https://www.deep-ml.com/problems/16
"""
transforming the range of values in your dataset so that different features contribute equally to the learning process

house_size in square feet: [900, 1100, 3000]
num_rooms: [2, 3, 4]
The model will favor house_size just because its numbers are bigger, even if num_rooms is equally important, so we need to normalise it

If features are on different scales, the cost function contours are elongated → gradient descent zigzags or takes forever to converge.
With scaling:
        •	Cost function contours are smoother
        •	Gradient descent moves cleanly downhill

standardization -> mean mean = 0, std dev = 1
x' = (x - mean)/std_dev # Useful when features have different distributions (e.g., Gaussian vs uniform).
min_max -> fixed range [0, 1]
x' = (x - x_min) / (x_max - x_min)


Scaling makes them all comparable.
"""

import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    standardized_data = np.round((data - data.mean(axis=0)) / data.std(axis=0), 4)
    normalized_data = np.round(
        (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)), 4
    )
    return standardized_data, normalized_data


def main():
    print(feature_scaling(data=np.array([[1, 2], [3, 4], [5, 6]])))


main()
