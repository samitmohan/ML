# https://www.deep-ml.com/problems/6
# 2x2 matrix
import math
import numpy as np


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    eigenvalues = []
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    disc = math.sqrt(trace**2 - 4 * det)
    eig1 = (trace + disc) / 2
    eig2 = (trace - disc) / 2
    eigenvalues.append(eig1)
    eigenvalues.append(eig2)

    return eigenvalues


def calcEV_usingNumpy(matrix):
    return list(np.linalg.eigvals(matrix).real)


def main():
    print(calculate_eigenvalues(matrix=[[2, 1], [1, 2]]))
    print(calcEV_usingNumpy(matrix=[[2, 1], [1, 2]]))


main()
