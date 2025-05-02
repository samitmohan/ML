# https://www.deep-ml.com/problems/10

import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    cov_matrix = np.cov(vectors, bias=False)  # Unbiased estimator (N-1)
    return cov_matrix.tolist()


def calcCov(vectors):
    def mean(vector):
        return sum(vector) / len(vector)

    m, n = len(vectors), len(vectors[0])  # m = features, n = observations
    means = [mean(feature) for feature in vectors]

    cov = [[0 for _ in range(m)] for _ in range(m)]

    for i in range(m):
        for j in range(m):
            cov_ij = sum( (vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n)) / (n - 1)
            cov[i][j] = cov_ij

    return cov



print(calcCov([[1, 2, 3], [4, 5, 6]]))
# Expected: [[1.0, 1.0], [1.0, 1.0]]

print(calcCov([[1, 2, 3], [6, 5, 4]]))
# Expected: [[1.0, -1.0], [-1.0, 1.0]]

print(calcCov([[1, 2, 1], [2, 4, 2]]))
# Expected: [[0.3333, 0.6666], [0.6666, 1.3333]] approx.