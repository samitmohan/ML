# https://www.deep-ml.com/problems/13
"""
Calculate determinant using Laplace Transformation

det(A)= (a11* ​C11) ​− (a12​ * C12)​ + (a13​* C13) ​− (a14 * ​C14)
where Cij​ = (−1)^i+j * det(Minor of aij​)
The process is recursive, breaking down the determinant calculation into smaller 3x3 determinants until reaching 2x2 determinants
    which are simpler to compute.

With numpy-:
det = np.linalg.det(A)
"""


def determinant_4x4(matrix: list[list[int | float]]) -> float:
    # first pick first row and col -> remaining goes back into the function
    # matrix[0][0] * cofactor(remaining matrix) - matrix[0][1] * cofactor(remaining matrix) + matrix[0][2] * cofactor(remaining matrix)...
    # returns matrix without ith row and jth column
    def minor(matrix, i, j):
        new_matrix = []
        for row_index, row in enumerate(matrix):
            if row_index == i:
                continue
            new_row = []
            for col_idx, val in enumerate(row):
                if col_idx == j:
                    continue
                new_row.append(val)
            new_matrix.append(new_row)
        return new_matrix

    # base case
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # laplace transformation
    total = 0
    for j in range(len(matrix[0])):
        sign = (-1) ** (0 + j)
        sub_minor = minor(matrix, 0, j)  # first row minors
        total += sign * matrix[0][j] * determinant_4x4(sub_minor)
    return total


def main():
    print(
        determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    )
    print(determinant_4x4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    # 2 * 4 * 3 * 6 = 144 (product of diagonal elements (since its lower trianglular matrix))
    print(determinant_4x4([[2, 3, 1, 5], [0, 4, 2, 1], [0, 0, 3, 7], [0, 0, 0, 6]]))


main()
