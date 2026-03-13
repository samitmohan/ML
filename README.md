# ML From Scratch

From-scratch implementations of machine learning algorithms, neural networks, and transformer architectures in Python. No high-level wrappers: every algorithm is built from NumPy operations or raw PyTorch modules to expose the underlying math.

## Highlights

**MicroGPT** - A complete GPT language model in 218 lines of pure Python. Includes custom autograd engine, multi-head attention, RMSNorm, and Adam optimizer. Trains on character-level data and generates text. No framework dependencies beyond `math` and `random`. See [`models/microgpt.py`](models/microgpt.py).

**Paper Implementations** - ResNet-34 built three ways (from scratch, PyTorch modules, HuggingFace), plus Rosenblatt's 1958 Perceptron. See [`papers/`](papers/).

**100+ Algorithm Implementations** - sourced from [Deep-ML](https://www.deep-ml.com/) problem sets:

| Category | Count | Examples |
|----------|-------|---------|
| Deep Learning | 29 | Autograd, CNN with backprop, LSTM, RNN, batch norm, residual blocks |
| Machine Learning | 36 | KNN, SVM, decision trees, PCA, k-means, logistic regression, naive bayes |
| Math / Linear Algebra | 28 | SVD, eigenvalues, Jacobi method, Newton's method, Lagrange optimization |
| Attention | 5 | Self-attention, multi-head attention, masked attention, positional encoding |
| Labs | 6 | Custom optimizer, training loop, transforms, loss functions |

## Repository Structure

```
algorithms/          104 implementations organized by topic
  attention/         Self-attention, MHA, positional encoding, layer norm
  deeplearning/      Autograd, CNN, LSTM, RNN, optimizers, loss functions
  machinelearning/   KNN, SVM, PCA, k-means, decision trees, regression
  math/              Linear algebra, calculus, matrix operations
  labs/              PyTorch training components (optimizer, loop, transforms)

models/              Neural network architectures
  microgpt.py        GPT from scratch in 218 lines
  mnist/             MNIST classification three ways (scratch, PyTorch, TensorFlow)
  karpathy/          Micrograd and Makemore exercises

papers/              Paper reproductions
  resnet/            Deep Residual Learning (He et al., 2015) - 3 implementations
  perceptron/        The Perceptron (Rosenblatt, 1958)

pytorch/             PyTorch building blocks and math foundations
numpy/               NumPy fundamentals and scientific computing
tests/               Test suite validating key implementations
```

## Setup

```bash
# Clone and install
git clone <repo-url> && cd ml
uv sync --group dev

# Run tests
uv run pytest

# Run any implementation directly
uv run python algorithms/deeplearning/autograd.py
```

## Approach

Every implementation prioritizes clarity over optimization. The goal is to understand the math behind each algorithm by implementing it from first principles, then comparing against established libraries (scikit-learn, PyTorch) where applicable. Most files include inline notes explaining the theory and link back to their source problem.
