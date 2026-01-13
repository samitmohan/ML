# https://www.deep-ml.com/problems/27

"""
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.

Example:
Input:

B = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        C = [[1, 2.3, 3],
             [4.4, 25, 6],
             [7.4, 8, 9]]

Output:

[[-0.6772, -0.0126, 0.2342],
                [-0.0184, 0.0505, -0.0275],
                [0.5732, -0.0345, -0.0569]]

Reasoning:

The transformation matrix P from basis B to C can be found using matrix operations involving the inverse of matrix C.

Notes

"""


import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    C_inv = np.linalg.inv(C)
    P = C_inv @ B
    return P


def main():
    B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    C = [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]
    print(transform_basis(B, C))


main()
