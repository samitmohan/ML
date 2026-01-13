# https://www.deep-ml.com/problems/46


import numpy as np
def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    precision = tp / (tp+fp)
    return precision

def main():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])

    result = precision(y_true, y_pred)
    print(result)

main()

