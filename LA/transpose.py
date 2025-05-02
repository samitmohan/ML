import numpy as np
def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Check if T and S are invertible (det non-zero)
    def is_invertible(matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * d - b * c
        return det != 0
    
    if not is_invertible(T) or not is_invertible(S): 
        return -1

    def inverse(matrix):
        # Inverse of 2x2 matrix: (1/det) * [d, -b, -c, a]
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * d - b * c
        return [[d / det, -b / det], [-c / det, a / det]]

    def matmul(M1, M2):
        # Matrix multiplication
        m, n = len(M1), len(M1[0])  # Dimensions of M1
        p, q = len(M2), len(M2[0])  # Dimensions of M2
        assert n == p, "Incompatible matrices for multiplication"
        
        mat = [[0.0 for _ in range(q)] for _ in range(m)]
        
        for i in range(m):
            for j in range(q):
                for k in range(n):
                    mat[i][j] += M1[i][k] * M2[k][j]
        return mat

    T_new = inverse(T)
    result = matmul(T_new, matmul(A, S))
    return result



def transform_matrix_better(A, T, S):
    if np.linalg.det(T) and np.linalg.det(S):
        T_inv = np.linalg.inv(T)
        return np.matmul(T_inv, np.matmul(A, S))
    return -1