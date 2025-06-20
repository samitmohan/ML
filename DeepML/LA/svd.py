import numpy as np

"""
Input:
a = [[2, 1], [1, 2]]

Output:
(array([[-0.70710678, -0.70710678], [-0.70710678,  0.70710678]]),
array([3., 1.]), array([[-0.70710678, -0.70710678], [-0.70710678,  0.70710678]]))

Reasoning:
U is the first matrix sigma is the second vector and V is the third matrix

notes on SVD -> pre-req, why, applications

Pre-requisites:
- Understanding of eigenvalues/eigenvectors
- Matrix multiplication, transpose
- Orthogonal matrices
- Square root, normalization

Why SVD?
- Decomposes any matrix (even non-square) into simpler parts
- Helps in compression, PCA, recommendation systems

Applications:
- Dimensionality reduction
- Image compression
- Latent semantic analysis (NLP)
- Solving ill-posed systems

SVD = U * Σ * V^T
- U and V are orthogonal matrices (rotations)
- Σ is a diagonal matrix with singular values (scales)

"""
# https://www.youtube.com/watch?v=mBcLRGuAFUk
# https://www.deep-ml.com/problems/12

# The SVD factors each matrix into an orthogonal matrix times a diagonal matrix  (the singular value)
#  times another orthogonal matrix: rotation times stretch times rotation.
# This question is only for 2 x 2 matrix which makes it easier


"""
import numpy as np
from math import sqrt

A = n * m records
A = U * Σ *  V^T
U is an (m × m) orthogonal matrix -> n * n (ev of A* A^T)
Σ is an (m × n) diagonal matrix of singular values
V is an (n ×n) orthogonal matrix


SVD effectively finds another orthonormal basis in our space and
 the representation of original matrix in that space.
So SVD gives us another point of view at the data, 
    where data is most distributed along (usually several) first axes

ui = normalise A
Helps understand the geometric transformation of any matrix

Rotate -> Scale/Stretch
Using numpy (full_matrices=False only gives minimal size orthogonal maitrices)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
"""


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_transpose = np.transpose(A)
    B = A_transpose @ A
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # eigenvector is orthogonal (symettric matriices have this prooperty)
    idx = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(eigenvalues[idx])  # diag values
    V = eigenvectors[:, idx]
    V_tranpose = V.T
    # Compute Ui from A @ Vi / singular_values
    U_columns = []
    for sigma, v_i in zip(singular_values, V.T):
        if sigma > 0:
            u_i = A @ v_i
            U_columns.append(u_i / sigma)
        else:
            # Orthogonal to the first column
            u_orth = np.array([-U_columns[0][1], U_columns[0][0]])
            U_columns.append(u_orth)

    U = np.column_stack(U_columns)  # build matrix from U_columns
    return U, singular_values, V_tranpose


if __name__ == "__main__":
    print(svd_2x2_singular_values([[2, 1], [1, 2]]))