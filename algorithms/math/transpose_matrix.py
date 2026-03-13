# https://www.deep-ml.com/problems/2
def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    # swap rows with columns
    n = len(a)  # row
    m = len(a[0])  # col
    b = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            b[j][i] = a[i][j]
    return b
