# https://www.deep-ml.com/problems/10
# https://www.youtube.com/watch?v=NVzxBz-rlhw&t=111s
import numpy as np
import matplotlib.pyplot as plt
'''
Notes on covariance

Corelation and covariance are very similary just that corelation is bounded between [-1,1] and covar isn't.

if both A and B are pos / neg -> covar is high
if A and B have opposite signs -> covar is low

Assume A = Apple, B = Banana and the happiness you get from eating them

          Apple    Banana
subject1    1         1 
subject2    3         0
subject3   -1        -1


Mean(Apple) = 1, Mean(Banana) = 0

Creating covar matrix for this (2*2)

A [first | second]
B [third | fourth]
  A         B


first = covar(A,A) which is nothing but variance(A)
fourth = covar(B,B) which is nothing but variance(B)
second = covar(A,B) = covar(B,A) = third

Covar(A, B) = E(AB) - E(A)(EB)  #E = Expectation = Mean
            = E(AB) - 0
      AB = 1
           0
           1
      E(AB) = 2/3
Covar(A,B) = Covar(B,A) = 2/3

Covar(A,A) = Variance = E(A^2) - E(A)^2 = (1+9+1)/3 - 1 = 8/3
Covar(B,B) = Variance = E(B^2) - E(B)^2 = (1+1)/3 - 0 = 2/3

Covariance Matrix is also symmetric (Transpose = Matrix)
Covar matrix = [8/3 2/3
                2/3 2/3]


Diagonal elements are variances, off diagonal elements are covar between variables
X = np.array([2.1, 2.5, 4.0, 3.6, 3.9])
Y = np.array([8, 10, 12, 11, 13])

def covariance(X, Y):
    n = len(X)
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n
    return sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n)) / (n - 1)

# Compute covariance matrix
cov_matrix = np.cov(X, Y)

How do you express covar matrx in a closed form way.

xi and xj are vectors
covar(i,j) = 1/n [sum(xi - xi') (xj - xj')]
For two variables X and Y:
Cov(X,Y) = (1/n−1) ∑(Xi−Xˉ)(Yi−Yˉ)



Person	Height (cm)	Weight (kg)
A	      160	        55
B	      170	        65
C	      180	        75

    Each column (Height, Weight) is a feature (aka variable, attribute).
    Each row (Person A, B, C) is an observation (aka data point, sample).

So:
    Features = number of columns
    Observations = number of rows
'''

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    cov_matrix = np.cov(vectors, bias=False)  # Unbiased estimator (N-1)
    return cov_matrix.tolist()


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    num_features = len(vectors) # features = rows
    num_observations = len(vectors[0]) # observations = columns
    def mean(v):
        return sum(v)/len(v)
    # calculating covariance
    means = [mean(feature) for feature in vectors]
    covar_matrix = [[0] * num_features for _ in range(num_features)]
    for i in range(num_features):
        for j in range(num_features):
            s = 0
            for k in range(num_observations):
                s += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
            covar_matrix[i][j] = s / (num_observations - 1)
    return covar_matrix

print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))
# Expected: [[1.0, 1.0], [1.0, 1.0]]
print(calculate_covariance_matrix([[1, 2, 3], [6, 5, 4]]))
# Expected: [[1.0, -1.0], [-1.0, 1.0]]
print(calculate_covariance_matrix([[1, 2, 1], [2, 4, 2]]))
# Expected: [[0.3333, 0.6666], [0.6666, 1.3333]] approx.

print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))
X = [1, 2, 3]
Y = [4, 5, 6]

plt.scatter(X, Y, color='blue')
plt.title("Feature X vs Feature Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show() # pos variance -> upwards line
