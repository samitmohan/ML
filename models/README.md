# Models

Neural network architectures implemented from scratch or using minimal framework code.

## MicroGPT

[`microgpt.py`](microgpt.py) - A complete GPT language model in 218 lines of pure Python. No PyTorch, no TensorFlow: just `math`, `random`, and a custom autograd engine built inline.

Includes:
- `Value` class with automatic differentiation (add, mul, pow, log, exp, relu)
- Topological sort for backpropagation through the computational graph
- Multi-head causal self-attention with KV caching
- RMSNorm, MLP blocks, residual connections
- Adam optimizer with linear learning rate decay
- Character-level tokenizer trained on name data

## MNIST

Three implementations of handwritten digit classification on the MNIST dataset:

- [`mnist/from_scratch/`](mnist/from_scratch/) - Pure NumPy. Network class with configurable layers, sigmoid activations, mini-batch SGD with backpropagation. Based on Michael Nielsen's neural networks book.
- [`mnist/pytorch/`](mnist/pytorch/) - PyTorch `nn.Module` with conv layers, dropout, and SGD.
- [`mnist/tensorflow/`](mnist/tensorflow/) - TensorFlow/Keras notebook implementation.

## Karpathy Exercises

Implementations following Andrej Karpathy's neural networks series:

- [`karpathy/micrograd/`](karpathy/micrograd/) - Micrograd from scratch: scalar-valued autograd engine and neural network training.
- [`karpathy/makemore/`](karpathy/makemore/) - Character-level language model using bigram statistics and neural approaches.
