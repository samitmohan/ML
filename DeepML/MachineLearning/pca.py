# https://www.deep-ml.com/problems/19

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

"""
notes
PCA's main goal is to find new, uncorrelated variables (principal components) that capture the maximum variance in the original data. The covariance matrix provides exactly the information needed to achieve this.
Goals : dimensionality reduction and feature extraction.
It transforms a set of possibly correlated variables into a set of linearly uncorrelated variables called principal components.
The core idea is to find directions (axes) in the data that capture the most variance.

Here's how it (Covariance Matrix) fits into the PCA process:
Capturing Relationships: 
The covariance matrix quantifies the linear relationships (or lack thereof) between all pairs of variables in your dataset. Variables with high covariance are strongly related, while those with low covariance are not.

Eigenvalue Decomposition: The core of PCA involves performing an eigenvalue decomposition (also known as eigendecomposition) on the covariance matrix.

Eigenvectors: The eigenvectors of the covariance matrix are the principal components. Each eigenvector represents a direction (a new axis) in the data space. These directions are orthogonal (perpendicular) to each other, meaning the principal components are uncorrelated.
Eigenvalues: The eigenvalues corresponding to each eigenvector represent the amount of variance captured by that particular principal component. A larger eigenvalue means that its corresponding eigenvector (principal component) explains more of the variance in the data.

Ordering Principal Components: By sorting the eigenvalues in descending order, you can rank the principal components from most important (capturing the most variance) to least important.

Dimensionality Reduction: You can then choose to keep only the top \(k\) principal components (those with the largest eigenvalues) to reduce the dimensionality of your data while retaining as much of the original variance as possible. This is the essence of dimensionality reduction in PCA.

What is Covariance?
Before understanding the covariance matrix, let's first understand covariance.
Covariance is a measure of the joint variability of two random variables. It tells us how much two variables change together.

Positive covariance: Indicates that the two variables tend to increase or decrease together.
Negative covariance: Indicates that when one variable increases, the other tends to decrease (and vice versa).
Zero covariance (or close to zero): Indicates that the two variables have no linear relationship.
Cov(X,Y) = sum [ (xi - xmean) * (yi - ymean) / n - 1 ]
Steps:
1. Standardise the data : subtract by mean and divide by std dev so that all data points are in the same locality.
2. Caluclate Covariance Matrix (quantifies the relationships (covariances) between all pairs of features in your dataset. It tells us how much two variables change together)
How: If you have p features, the covariance matrix will be a p * p symmetric matrix where:

Diagonal elements are the variances of each feature.
Off-diagonal elements are the covariances between feature Xi and feature Xj

3. Calculate eig val and eig vectors
Av=λv (A represents cov matrix) , det(A−λI)=0 {calculating eigenvalues}
covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.
Eigenvectors represent the directions (the principal components), and eigenvalues represent the magnitude or amount of variance along those directions.

4.Sort Eigenvalues and Select Principal Components
"""


def pca(data: np.ndarray, k: int) -> np.ndarray:
    # step1 : standardise data
    mean = np.mean(data, axis=0)  # rows {axis=1 is columns}
    std_dev = np.std(data, axis=0)  # rows
    std_dev[std_dev == 0] = 1.0  # cases where std_dev = 0 to avoid div by 0
    scaled_data = (data - mean) / std_dev
    # step2 : cov matrix
    cov_matrix = np.cov(scaled_data.T)  # .cov expects observations as columns
    # step3 : ev and vectors
    eigenval, eigenvectors = np.linalg.eig(cov_matrix)
    # step4 : sort based highest and choose k
    sorted_idx = np.argsort(eigenval)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    principal_components = sorted_eigenvectors[:, :k]
    return np.round(principal_components, 4)


# using sci-kit learn


def pca_with_skicit(data, k):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca_model = PCA(n_components=k)
    pca_model.fit(scaled_data)
    # principal_components = pca_model.fit_transform(scaled_data) returns transformed data (the original samples projected onto the new principal components).
    principal_components_eigenvectors = pca_model.components_.T[:, :k]
    return np.round(principal_components_eigenvectors, 4)


def main():
    data, k = np.array([[1, 2], [3, 4], [5, 6]]), 1
    # print(pca(data, k))
    print(pca_with_skicit(data, k))
    # [[0.7071]
    # [0.7071]]


main()
