import numpy as np


class TestKNN:
    def test_iris_classification(self):
        from sklearn.datasets import load_iris
        from algorithms.machinelearning.knn import KNNClassifier

        iris = load_iris()
        X, y = iris.data, iris.target

        # Use a simple train/test split
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        X_train, y_train = X[indices[:120]], y[indices[:120]]
        X_test, y_test = X[indices[120:]], y[indices[120:]]

        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.9, f"KNN accuracy {accuracy:.2f} too low on Iris"

    def test_euclidean_distance(self):
        from algorithms.machinelearning.knn import KNNClassifier

        knn = KNNClassifier()
        d = knn.euclid_dist(np.array([0, 0]), np.array([3, 4]))
        assert abs(d - 5.0) < 1e-6


class TestLinearRegression:
    def test_normal_equation(self):
        from algorithms.machinelearning.linear_regression import linear_regression_normal_equation

        # y = 2x + 1 with bias column
        X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
        y = [3, 5, 7, 9, 11]
        theta = linear_regression_normal_equation(X, y)
        # theta should be [1, 2] (intercept=1, slope=2)
        assert abs(theta[0] - 1.0) < 1e-3
        assert abs(theta[1] - 2.0) < 1e-3

    def test_gradient_descent(self):
        from algorithms.machinelearning.linear_regression import linear_regression_gradient_descent

        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
        y = np.array([3, 5, 7, 9, 11])
        theta = linear_regression_gradient_descent(X, y, alpha=0.01, iterations=1000)
        assert abs(theta[0] - 1.0) < 0.5
        assert abs(theta[1] - 2.0) < 0.5


class TestPCA:
    def test_output_shape(self):
        from algorithms.machinelearning.pca import pca

        data = np.random.randn(50, 5)
        components = pca(data, k=2)
        assert components.shape == (5, 2)

    def test_orthogonality(self):
        from algorithms.machinelearning.pca import pca

        np.random.seed(42)
        data = np.random.randn(100, 4)
        components = pca(data, k=2)
        # Principal components should be approximately orthogonal
        dot = np.abs(np.dot(components[:, 0], components[:, 1]))
        assert dot < 0.1, f"Components not orthogonal: dot product = {dot}"


class TestKMeans:
    def test_separable_clusters(self):
        from algorithms.machinelearning.k_means import k_means_clustering

        # Two clearly separable clusters
        points = [(0, 0), (1, 0), (0, 1), (10, 10), (11, 10), (10, 11)]
        centroids = k_means_clustering(
            points=points,
            k=2,
            initial_centroids=[(0, 0), (10, 10)],
            max_iterations=100,
        )
        # Centroids should be near (0.33, 0.33) and (10.33, 10.33)
        centroids_sorted = sorted(centroids, key=lambda c: c[0])
        assert centroids_sorted[0][0] < 2.0
        assert centroids_sorted[1][0] > 8.0
