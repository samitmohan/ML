# https://www.deep-ml.com/problems/8
import numpy as np


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    def is_invertible(matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * d - b * c
        return det != 0

    if not is_invertible(matrix):
        return -1
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]


def inv(matrix):
    return np.linalg.inv(matrix)
