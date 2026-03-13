# K Nearest Neighbours 
'''
KNN is a non-parametric, lazy learning algorithm used for both classification and regression. The principle is simple:

Find the K closest data points (neighbors) to a new, unseen data point.
For classification: Assign the new point to the class that is most common among its K neighbors (majority vote).
For regression: Assign the new point the average (or median) of the target values of its K neighbors.
classification is the most common use case.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import numpy as np
from collections import Counter
class KNNClassifier:
    def __init__(self, k=3) -> None:
        self.k = k
        self.x_train, self.y_train = None, None

    def fit(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")
        self.x_train = x
        self.y_train = y
        print(f"KNNClassifier fitted with {x.shape[0]} samples and {x.shape[1]} features.")

    def euclid_dist(self, x1, x2):
        return np.sqrt(np.sum(x1-x2)**2)

    def predict_new(self, x_new):
        dist = [self.euclid_dist(x_new, x_train) for x_train in self.x_train]
        k_idx = np.argsort(dist)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0] # returns label (element)
    
    def predict(self, x_new):
        predictions = [self.predict_new(x) for x in x_new]
        return np.array(predictions)
    

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Feature names: {feature_names}")
print(f"Target names: {target_names}")
print("\nSample data:")
print(X[:5])
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nPredictions vs Actual:")
for i in range(min(10, len(y_pred))): 
    print(f"Predicted: {target_names[y_pred[i]]} (ID: {y_pred[i]}), Actual: {target_names[y_test[i]]} (ID: {y_test[i]})")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on the test set: {accuracy:.4f}")
