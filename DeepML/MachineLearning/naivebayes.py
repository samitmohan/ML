# https://www.deep-ml.com/problems/140


import numpy as np
'''
Laplace smoothing parameter (default: 1.0) to handle zero probabilities.

'''
class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        # Computes class priors (P(C) and likelihoods (Bayes theorem))
        # Fit model to binary features X and labels y
        self.classes, class_count = np.unique(y, return_counts=True)
        self.priors = {
                cls: np.log(class_count[i] / len(y)) for i, cls in enumerate(self.classes)
        }
        self._fit_bernouli(X, y)


    def _fit_bernouli(self, X, y):
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y==cls]
            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (X_cls.shape[0] + 2 * self.smoothing)
            self.likelihoods[cls] = np.log(prob), np.log(1-prob)

    def _compute_bayesion(self, sample):
        bayesion_prob = {}
        for cls in self.classes:
            probability = self.priors[cls]
            prob_1, prob_0 = self.likelihoods[cls]
            likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)
            probability += likelihood 
            bayesion_prob[cls] = probability
        return max(bayesion_prob, key=bayesion_prob.get)


    def predict(self, X):
        # Computes probability of a feature being present (1) or absent (0) in a given class.
        # Predict class labels for test set X
        return np.array([self._compute_bayesion(sample) for sample in X])
