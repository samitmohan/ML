import math
# https://www.deep-ml.com/problems/22

# Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.


def sigmoid(z: float) -> float:
    # Your code here
    sigm = 1 / (1 + math.exp(-z))
    result = round(sigm, 4)
    return result

def main():
    print(sigmoid(z = 0))
main()
