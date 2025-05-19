"""
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

# The SVD factors each matrix into an orthogonal matrix times a diagonal matrix  (the singular value) times another orthogonal matrix: rotation times stretch times rotation.
# This question is only for 2 x 2 matrix which makes it easier

"""
Steps-:
1) Find eigenvalues and eigenvectors of A^T . A = lambda1, lambda2,  and x (eigenvector)
2) Normalise x -> x / root(elements) for all elements of x
3) V Matrix = [normalised eigenvectors], Find V^T
4) Order of sigma = Order of A, sigma1 = root(lambda1) and sigma2 = root(lambda2) {we only care about non zero vals} : Diagonal matrix form = diagonal entries are sigma1 and sigm2
5) Find U : A*A^T : Same process for V
6) Return product of U * Sigma * V^T
"""
import numpy as np
from math import sqrt


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    def inverse_of_matrix(A):
        # to be used in sigma inv calc
        pass

    At = np.transpose(A)
    At_dot_A = np.dot(A, At)
    eigenvalues, eigenvectors = np.linalg.eigh(At_dot_A)
    # figure this out
    # norm = sqrt(sum(eigenvectors[i][j]) for i in range(len(eigenvectors) for j in range(len(eigenvectors[0]))))
    # A_normalized = [[eigenvectors[i][j]/norm for j in range(len(eigenvectors[0]))] for i in range(len(eigenvectors))]
    # V_transpose = np.transpose(A_normalized)
    # print(V_transpose)
    V = np.linalg.norm(eigenvectors)
    print(V)
    V_transpose = np.transpose(np.linalg.norm(eigenvectors))
    print(V_transpose)
    # same for U

    # finding order of sigma
    sigma1, sigma2 = sqrt(eigenvectors)
    # create matrix whose diagonal elements are sigma1 and sigma2
    singular_values = np.sqrt(
        np.maximum(eigenvalues, 0)
    )  # Avoid negative roots due to float errors
    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, singular_values)
    np.zeros_like(Sigma).T
    # calc sigma inverse
    # Step 5: Find U = A * V * Sigma_inv
    U = np.dot(A, np.dot(V, Sigma_inv))
    SVD = []
    SVD.append([U, singular_values, V])


# return SVD


def main():
    svd_2x2_singular_values(A=[[2, 1], [1, 2]])


main()
"""

Input:
a = [[2, 1], [1, 2]]

Output:
(array([[-0.70710678, -0.70710678], [-0.70710678,  0.70710678]]),
array([3., 1.]), array([[-0.70710678, -0.70710678], [-0.70710678,  0.70710678]]))

Reasoning:

U is the first matrix sigma is the second vector and V is the third matrix
"""
