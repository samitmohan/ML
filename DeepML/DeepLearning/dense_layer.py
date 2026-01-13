# https://www.deep-ml.com/problems/40

'''
X → shape (batch_size, d_in)
W → shape (d_in, d_out)
b → shape (1, d_out)
Y → shape (batch_size, d_out)

accum_grad = ∂Y / ∂L​

self.layer_input = X # stores the exact input used during the forward pass.  gradients of weights depend on what input produced the output.

grad_W = self.layer_input.T @ accum_grad
| Term            | Shape                 |
| --------------- | --------------------- |
| `layer_input.T` | `(d_in, batch_size)`  |
| `accum_grad`    | `(batch_size, d_out)` |
| Result          | `(d_in, d_out)` ✔     |

Bias is added once per output neuron
Every sample contributes to the same bias
grad_b = np.sum(accum_grad, axis=0, keepdims=True)
| Term          | Shape                 |
| ------------- | --------------------- |
| `accum_grad`  | `(batch_size, d_out)` |
| Result        | `(1, d_out)` ✔       |

self.W = self.optimizer.update(self.W, grad_W)
self.bias = self.optimizer.update(self.bias, grad_b)
W_new = W - lr * grad_W
b_new = b - lr * grad_b

Forward:
X ──► Dense ──► Y

Backward:
dL/dY ──► grad_W, grad_b
        └─► dL/dX (returned)

'''

import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
    
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		""" Return the shape of the output produced by the forward pass, which should be (self.n_units,) """
		return self.output_shape

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit,
                                   (self.input_shape[0], self.n_units))
        self.bias = np.zeros((1, self.n_units))
        self.output_shape = (self.n_units,)
        self.optimizer = copy.deepcopy(optimizer)

    def parameters(self):
        return self.input_shape[0] * self.n_units + self.n_units

    def forward_pass(self, X):
        self.layer_input = X
        return X @ self.W + self.bias

    def backward_pass(self, accum_grad):
        # Gradient w.r.t input
        grad_input = accum_grad @ self.W.T

        if self.trainable:
            # Gradients
            grad_W = self.layer_input.T @ accum_grad
            grad_b = np.sum(accum_grad, axis=0, keepdims=True)

            # Update
            self.W = self.optimizer.update(self.W, grad_W)
            self.bias = self.optimizer.update(self.bias, grad_b)

        return grad_input


# Expected input
# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)


# Expected output:
'''
Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
Backward pass output: [[ 0.00129588  0.00953634]]

My output is
Forward pass output: [[ 0.10162127 -0.33551992 -0.64490545]]
Backward pass output: [[ 0.20816524 -0.22928937]]

'''