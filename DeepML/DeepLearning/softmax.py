# https://www.deep-ml.com/problems/23
# Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

'''
The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.
softmax(z) = exp(z) / sum(exp(z))
'''

import math

def softmax(scores: list[float]) -> list[float]:
	# Your code here
	sum_exp = 0
	for i in scores:
		sum_exp += math.exp(i)
	probabilities = []
	for j in scores:
		probabilities.append(math.exp(j) / sum_exp)
	return probabilities

def main():
    scores = [1, 2, 3]
    print(softmax(scores)) 
    # [0.0900, 0.2447, 0.6652]

main()