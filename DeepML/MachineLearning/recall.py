# https://www.deep-ml.com/problems/52
import numpy as np
def recall(y_true, y_pred):
    # recall is just tp / tp + fn, ability to understand how many classes a model can classify
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1)) 
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    res = tp / (tp+fn)
    return round(res, 3)