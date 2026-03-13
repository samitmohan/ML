from collections import Counter

def confusion_matrix(data):
    """
    Generate confusion matrix for binary classification.

    data contains (y_pred, y_true)
    Returns [[TN, FP], [FN, TP]]
    """
    TN = FP = FN = TP = 0

    for y_pred, y_true in data:
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1

    return [[TN, FP], [FN, TP]]

