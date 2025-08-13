# accuracy_score = num_correct_pred/total_pred

# https://www.deep-ml.com/problems/36
'''
Write a Python function to calculate the accuracy score of a model's predictions. 
The function should take in two 1D numpy arrays: y_true, which contains the true labels, 
and y_pred, which contains the predicted labels. 
It should return the accuracy score as a float.

'''
import numpy as np

def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays y_true and y_pred must have the same length.")
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def main():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output) # 0.8333333333333334

main()