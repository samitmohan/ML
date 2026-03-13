# Logistic Regression
# https://www.deep-ml.com/problems/104

'''
https://aman.ai/primers/ai/linear-logistic-regression/#how-logistic-regression-works-a-practical-example
Input: 
predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0)

X : [1,1], [2,2], [-1,-1], [-2,-2] These are the data points
Weights : [1, 1]
Bias : 0

y = mx + c
y = x dot weights + bias
apply sigmoid to y, if > 0.5 -> 1 else 0
Output: [1 1 0 0]
'''

import numpy as np

def predict(X, weights, bias):
    forward_pass = np.dot(X, weights) + bias
    sigmoid = 1 / (1 + np.exp(-forward_pass))
    threshold = .5
    return np.round(sigmoid >= threshold).astype(int)

def main():
    print(predict(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0))
main()



