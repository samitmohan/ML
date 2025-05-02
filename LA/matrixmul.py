import numpy as np
def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    m,n1 = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    if n1!=n2: return -1
    # else do matrix mul
    c = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n1):
                c[i][j] += a[i][k] * b[k][j]
    return c

def matrixmulNumpy(a, b):
    a, b = np.array(a), np.array(b)
    if a.shape[1] != b.shape[0]: return -1
    return (a @ b).tolist()
    # return np.matmul(A,B).tolist()