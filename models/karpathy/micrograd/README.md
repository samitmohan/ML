# Learnings

- What are neural nets? Math expressions that take input as data (weights, parameters) : forward pass -> loss function (measures accuracy of predictions : low when network is behaving well) -> backpropagation to get gradient and tune parameters to minimise loss function (this is called gradient descent)
- All of this is available in PyTorch, but how to do this manually? .

### TensorFlow

- In TensorFlow, tensors are like arrays, building blocks for rep + manipulating data. {can hold any data type, immutable, multi-dim, graphs}

scalar_tensor = tensorflow.constant(42, dtype=tf.int32)
matrix_tensor = tensorflow.constant([1.0, 2.0], [3.0, 4.0], dtpye=tf.float32)
ans = scalar_tensor + matrix_tensor

## Pytorch

- Uses dynamic computation : graph is built on the fly as operations are performed. (easier to debug (can use control flow constructs))
- Provides tensor = numpy with GPU acceleration.
- Dynamic NN
- Autograd : built in atuomatic differentation library. Computies gradients for tensors making to easy to perform backpropagation fo training neural networks.

### Topic : What is autograd?

- It is a mathematical technique used for automatically and efficiently computing gradients (derivatives) of mathematical expressions.
- Optimises NN models during training.

#### How it works?

1. Computation Graph

- Operates on computational graph : operations and dependencies between variables in math expression.
- Each operation (addition/subtraction/activation function) is a node in computation graph and variables (model parameters, input data) are edges.

2. Forward Pass

- Input data is processed through neural network and results are computed.
- Applies model's operation to input data layer by layer to predict output.

3. Backpropagation

- Autograd is used to automatically compute gradients wrt model parameters (calculate loss function etc..)
- Gradients = rate of change of function wrt input variables. (How sensitive loss function is to changes in parameters)
- Reset gradients (0) and calculate loss function again.

4. Chain Rule

- Autograd leverages chain rule to compute gradients. Derivative of composite function = product of derivatives of indiviual components.

5. Optimization

- Once gradients are computed they are used in opt algorithms like adam/gradientdescent.
- Adjusts parameters to minimise the loss function + trains NN.

### More on Autograd (docs on pytorch.org)

- autograd records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the
  input tensors and roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

- The gradient computation using Automatic Differentiation is only valid when each elementary function being used is differentiable. Unfortunately many of
  the functions we use in practice do not have this property (relu or sqrt at 0, for example).

- Can save the context using ctx.save_for_backward(...) when doing forward pass to save for backpropagation (reference). Tensor Parameters = data, grad, grad_fn, is_leaf, requires_grad

- Every grad function points to some node.

- Video Explanation = https://www.youtube.com/watch?v=MswxJw-8PvE

# What is micrograd?

A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API.
Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values.

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b # child nodes are a and b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass


# initialise backprop at node g (leaf -> root (reverse chain rule : calculate derivative/descent of g wrt all internal nodes and inputs))
# Then we can calculate derivatives with respect to inputs : How a and b are affecting g

g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db')')')
```

- Value object behaves like Tensor (Scalar values wrapped in an object)
- All operations on Tensor can be done in parallel (since they are arrays) and makes computation faster.

### Files

engine.py & nn.py
Backpropagation etc : engine.py
Define Neuron, Layer, MLP, Value etcc : nn.py
