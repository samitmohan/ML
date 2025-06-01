# Notes from Karpathy Lectures + Basics

**PyTorch**: Deep Learning Framework which accelerates the model training process.  
Built by Facebook.

**Tensors**: Multi-dimensional Arrays (think of them as 5D+ objects).

Building a neural network using PyTorch.

Deep Learning represents prediction (generalizes rules of operation: correct behaviour).

**Philosophy**: Rules do not matter, data does.

Uses ANN (Artificial Neural Networks):  
Input → Hidden → Output, using backpropagation / gradient descent.

Thousands of samples are required for training → Train → Predict.

Can use pre-trained models and then fine-tune them as per our liking → make predictions → PyTorch.

Built around the tensor class, has high performance on GPUs with CUDA, built-in backpropagation (autograd), and neural network building blocks (Python + C++).

---

## Adam Optimizer

Adam Optimizer: An algorithm in deep learning that helps adjust parameters of a neural network in real time to improve accuracy and speed.  
**Adam** stands for Adaptive Moment Estimation.  
It helps neural networks learn faster and converge quickly towards the optimal set of parameters that minimize the loss function.

One of the main advantages of Adam is its ability to handle noisy and sparse datasets, which are common in real-world applications.

---

## Gradient Descent

Gradient descent is an optimization algorithm commonly used in machine learning and deep learning to minimize a cost or loss function. Its primary purpose is to find the optimal parameters (weights and biases) for a model that minimizes the error between predicted and actual values.

**Steps:**
1. Calculate the gradient (derivative) of the cost function with respect to each parameter.
2. Update parameter:  
   `parameter -= learning_rate * gradient(parameter)`
3. Learning rate = hyperparameter that controls the size of each step.
4. Repeat these two steps until the cost function converges to a minimum value.

**Stochastic Gradient Descent (SGD):**  
Instead of using the entire dataset to compute the gradient at each iteration, it uses a single random data point (or a small batch of data) to estimate the gradient.

**Adam** is an extension to the GD algorithm.

---

## Neural Network Layers

- **Input Layer**: Accepts raw data.
- **Conv2D Layer**: Processes grid-like data (images) → extracts features from input data (filters).
- **Activation Layer (ReLU)**: Applies a simple threshold function (detects patterns in data).
- **Pooling Layer**: Downsamples the feature maps produced by convolutional layers.  
  (e.g., Max pooling selects the max value from a group of values in a local region of the input.)
- **Flatten Layer**: Converts the multi-dimensional output of the previous layer into a 1D vector.
- **Fully Connected Layer (Linear)**: Connects every neuron from the previous layer to every neuron in the current layer (final decision-making).
- **Dropout Layer**: Prevents overfitting by randomly turning off some neurons during training.
- **Batch Normalization**: Normalizes the input to a layer in order to stabilize and speed up training.
- **Softmax Layer**: Computes probabilities of each class; outputs sum to 1.

---

## Cross Entropy

"Cross-entropy" is a commonly used loss function in machine learning and deep learning, particularly for classification tasks.  
It measures the dissimilarity or "distance" between the predicted probabilities (or scores) and the true labels of the data.

---

## Learnings from Neural Network + Deep Learning Chapter (Data Science for Python, basics)

### Neural Network

- **Perceptrons**: The simplest type of artificial neuron. Takes several binary inputs, applies weights, sums them, and passes the result through an activation function (like a step or sigmoid).
- **Activation Functions**: Functions like sigmoid, tanh, or ReLU that introduce non-linearity, allowing the network to learn complex patterns.
- **Forward Propagation**: The process of passing input data through the network to get predictions.
- **Backpropagation**: The process of calculating gradients and updating weights to minimize the loss.
- **Layers**: Stacking perceptrons into layers (input, hidden, output) allows the network to learn hierarchical representations.

### Deep Learning

- **Deep** = Many hidden layers (not just one).
- **Representation Learning**: The network learns the best features for the task directly from data.
- **Generalization**: The goal is not just to memorize the training data, but to perform well on new, unseen data.
- **Transfer Learning**: Using a model trained on one task as the starting point for a related task (fine-tuning).

---

## Summary

- PyTorch makes building, training, and experimenting with neural networks fast and flexible.
- Deep learning is all about learning from data, not hand-coding rules.
- Modern neural networks use layers, activation functions, and optimizers like Adam to learn complex patterns from large datasets.
- Understanding the basics (perceptrons, gradient descent, backpropagation) is key to mastering deep learning.

---

