# pathetic how i don't know how to inverse a matrix
"""
inv(A) * A = Identity Matrix
this let's us know how inv(A) is defined.
det(A) != 0 for inverse (since det(0) is just flat space) means it “flattens” n-dimensional space into a lower-dimensional space
    either no solutions (inconsistent) or infinitely many solutions (underdetermined)
A^-1 = 1/det(A) {adj(A)} -> adj(A) is transpose of its cofactor matrix (C_ij = (-1)^i+j & Minor_ij)
each cofactor represents how much area changes when you remove one coordinate direction
adj(A) packages up all information needed to undo A upto scale of det(A)
"""

# 2x2


# direct -> cramer's rule for 2x2 since minors are just inverse of elements (-b and -c)
def inverse_2x2(matrix):
    det_A = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    print(det_A)
    # inverse_matrix = adj(A) / det_A
    if det_A == 0:
        print("Singular Matrix has no inverse")
    # adj entries = [d,,-b,,-c,,a]
    new_matrix = [matrix[1][1], -matrix[0][1], -matrix[1][0], matrix[0][0]]
    inv = []
    for elem in new_matrix:
        compute = elem // det_A
        inv.append(compute)
    return inv


# 3x3


def inverse_3x3(matrix):
    # calculate det, build cofactor matrix (sign * minor), form adj (transpose(cofactor)), divide by det
    def minor(matrix, i, j):
        new_matrix = []
        for row_idx, row in enumerate(matrix):
            new_row = []
            if row_idx == i:
                continue
            for col_idx, col in enumerate(row):
                if col_idx == j:
                    continue
                # otherwise -> add this to new_matrix
                new_row.append(matrix[row_idx][col_idx])
            new_matrix.append(new_row)
        return new_matrix

    def det(matrix):
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        # 3x3
        return (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )

    # forming cofactor matrix
    cofactor_matrix = [
        [(-1) ** (i + j) * det(minor(matrix, i, j)) for j in range(3)] for i in range(3)
    ]
    # adjoint = transpose of cofactor_matrix
    adj = [list(row) for row in zip(*cofactor_matrix)]
    det_matrix = det(matrix)
    inverse = [[adj[i][j] / det_matrix for j in range(3)] for i in range(3)]
    return inverse


# general (gauss jordan method) / LU decomposition (same complexity)
# the big guns (O(n^3))
"""
intuition -> [matrix | identity matrix] -> perform bunch of magic operations to make -> [identity matrix | inverse]
if you can't produce identity matrix on left side -> matrix isn't a square matrix
magic operations = row reduction
matrix = 2x2 -> transformed to 2x4
[1 3 | 1 0 
 2 5 | 0 1] 
 after row reduction (right side is inverse of original matrix)
[1 0 | -5 3
 0 1 | 2 -1 ]
"""


def inverse_general(matrix):
    """
    transform matrix n*n into n * 2n by adding identity matrix to right side
    perform row reduction until all elements except pivot elements are 0
        Find a non-zero pivot in column i (swap rows if needed).
        Scale the row so that the pivot entry becomes 1.
        Eliminate that column in all other rows by subtracting multiples of the pivot row
    extract right block = inv(A)
    """
    rows, n = len(matrix), len(matrix[0])
    if rows != n:
        raise ValueError("non square matrix has no inverse fool")

    # transform into augmented matrix
    def augmented(matrix):
        aug = []
        for i, row in enumerate(matrix):
            identity_row = [1 if j == i else 0 for j in range(rows)]
            aug.append(row + identity_row)
        return aug

    aug = augmented(matrix)

    # perform row reduction
    # helper functions
    def swap_rows(aug, i, k):
        aug[i], aug[k] = aug[k], aug[i]

    def scale_row(aug, i, factor):
        aug[i] = [x / factor for x in aug[i]]

    def eliminate_column(aug, pivot_row, target_row, col):
        factor = aug[target_row][col]
        aug[target_row] = [
            xj - factor * xi for xj, xi in zip(aug[target_row], aug[pivot_row])
        ]

    # reduction
    for i in range(rows):
        for k in range(i, rows):
            if aug[k][i] != 0:
                # pivot elem found
                # swap to topmost posn
                swap_rows(aug, i, k)
                break
        # divide / nomalise entire row by pivot so aug[i][i] becomes 1 (scales every entry in row i)
        scale_row(aug, i, aug[i][i])
        # for every other row where i != j
        # we need to make aug[j][i] = 0 {reduction} by subtracting it from factor of pivot row
        for j in range(rows):
            if j != i:
                eliminate_column(aug, i, j, i)

    # left half of aug will be the identity, the right half will be A^-1 : [Iₙ | A⁻¹]
    # We only want the right part -> skip columns and after that -> grab matrix
    # inverse = [n,2n).
    inv = [row[n:] for row in aug]
    return inv


def main():
    matrix_2x2 = [[2, 1], [1, 1]]
    matrix_3x3 = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    # print(inverse_2x2(matrix_2x2))
    print(inverse_3x3(matrix_3x3))
    # print(inverse_general(matrix_2x2))
    print(inverse_general(matrix_3x3))


main()
