# https://www.deep-ml.com/problems/28

"""
Input:

A = [[-10, 8],
         [10, -1]]

Output:

(array([[  0.8, -0.6], [-0.6, -0.8]]),
    array([15.65247584,  4.47213595]),
    array([[ -0.89442719,  0.4472136], [ -0.4472136 , -0.89442719]]))

Reasoning:

The SVD of the matrix A is calculated using the eigenvalues and eigenvectors of A^T A and A A^T. The singular values are the square roots of the eigenvalues, and the eigenvectors form the columns of matrices U and V.
"""

import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    t1 = x1 / h1
    t2 = x2 / h2

    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    V = np.diag(1.0 / s) @ U.T @ A

    return U, s, V
