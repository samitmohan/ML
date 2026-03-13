# https://www.deep-ml.com/problems/61
import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True / False Positives / Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Precision & Recall (with safety)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta2 = beta ** 2

    # F-beta score
    if precision + recall == 0:
        return 0.0

    fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)

    return round(fbeta, 3)

