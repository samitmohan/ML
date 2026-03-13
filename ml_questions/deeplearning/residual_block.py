# https://www.deep-ml.com/problems/113
'''
Implement a function that creates a simple residual block using NumPy. 
The block should take a 1D input array, process it through two weight layers (using matrix multiplication), apply ReLU activations, and add the original input via a shortcut connection before a final ReLU activation.

Input:
x = np.array([1.0, 2.0]), w1 = np.array([[1.0, 0.0], [0.0, 1.0]]), w2 = np.array([[0.5, 0.0], [0.0, 0.5]])

Output:
[1.5, 3.0]

The input x is [1.0, 2.0]. 
First, compute w1 @ x = [1.0, 2.0], apply ReLU to get [1.0, 2.0]. 
Then, compute w2 @ [1.0, 2.0] = [0.5, 1.0]. 
Add the shortcut x to get [0.5 + 1.0, 1.0 + 2.0] = [1.5, 3.0]. Final ReLU gives [1.5, 3.0].


'''
import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    def relu(x):
        return np.maximum(0, x)

    conv1 = w1 @ x
    op1 = relu(conv1)
    conv2 = w2 @ op1
    op2 = relu(conv2)

    # add shortcut to op2
    op2 += x
    return relu(op2)


def main():
    x = np.array([1.0, 2.0]) 
    w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
    print(residual_block(x, w1, w2))

main()