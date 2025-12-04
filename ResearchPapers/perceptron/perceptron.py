# Implementing the perceptron paper

"""
Source: https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf
Dated year 1958, University of Pensylvannia
Written by F Rosenblatt
NOTE:
    Draw-backs of the perceptron :
    1. Does not work on datasets where the data is not linearly seperable
    2. Assumes that the decision boundary will always be linear on the training data and while making predictions.

    The perceptron typically updates weights one sample at a time (stochastic gradient descent like).
    We need to loop over each sample individually within the iteration loop.
"""

import numpy as np


class Perceptron:
    def __init__(self, threshold=0.0, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.threshold = threshold

    def activation_fn(self, z):
        return np.where(z > self.threshold, 1, 0)

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

        for epoch in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X_train):
                z = np.dot(x_i, self.weights) + self.bias
                y_hat = self.activation_fn(z)

                # Perceptron learning rule (only update on misclassification)
                if y_hat != y_train[idx]:
                    update = self.learning_rate * (y_train[idx] - y_hat)
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            # Optional convergence check (stops early if no errors)
            if errors == 0:
                print(f"Training converged at epoch {epoch}")
                break

        return self.weights, self.bias

    def predict(self, X_test):
        X_test = np.array(X_test)
        z_test = np.dot(X_test, self.weights) + self.bias
        return self.activation_fn(z_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        accuracy = np.mean(y_pred == y_test)

        TP = np.sum((y_test == 1) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print("Evaluation Metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1_score:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate

    model = Perceptron(threshold=0.0, learning_rate=0.1, n_iterations=100)
    model.fit(X, y)
    preds = model.predict(X)
    print("Predictions:", preds)
    model.evaluate(X, y)

