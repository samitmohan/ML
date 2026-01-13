# https://www.deep-ml.com/problems/66?from=Linear%20Algebra
import numpy as np
def orthogonal_projection(v, L):
    v, L = np.array(v, dtype=float), np.array(L, dtype=float)
    projection = (np.dot(v, L) / np.dot(L, L)) * L
    return np.round(projection, 3).tolist()

    


def main():
    v = [3, 4]
    L = [1, 0]
    print(orthogonal_projection(v, L))

main()
