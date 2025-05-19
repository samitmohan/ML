# https://www.deep-ml.com/problems/11
"""
notes

Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b.
The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x

Matrix should be diagonal dominant where diag elem a11, a22 abs val > all non diagonal elements # by interchanging pos of row and col
This method assumes that all diagonal elements of ( A ) are non-zero and that the matrix is diagonally dominant or properly conditioned for convergence.

"""

import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    """
    each equation x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i))
    diagonal elements = a[i][i], non diagonal elements = a[i][j] where i!=j
    xi^(k+1) = 1/a[i][i](b[i] - sum(a[i][j] * x[j]^k))
    inital guess = x = [0,0,0,0,0,0,0,...0]
    k iterations
    """
    size = len(b)  # x y z
    x = [0.0] * size  # ans
    for _ in range(n):
        soln = []
        for i in range(size):
            summ = sum(A[i][j] * x[j] for j in range(size) if j != i)
            new_xi = round((b[i] - summ) / A[i][i], 4)
            soln.append(new_xi)
        # update x
        x = soln
    return x


def solve_jacob_using_numpy(A, b, n):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    D = np.diag(A)  # diagonal elements
    ND = A - np.diagflat(D)  # non-diagonal elements
    x = [0.0] * len(b)
    for i in range(n):
        x = (b - np.dot(ND, x)) / D
        x = np.round(x, 4)
    return x.tolist()


def main():
    # print(solve_jacobi(A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2))
    print(
        solve_jacob_using_numpy(
            A=[[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b=[-1, 2, 3], n=2
        )
    )


main()
