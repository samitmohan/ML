# https://www.deep-ml.com/problems/130
import numpy as np


'''
Create a function that trains a basic Convolutional Neural Network (CNN) using backpropagation. 
The network should include one convolutional layer with ReLU activation, followed by flattening and a dense layer with softmax output, and a cross entropy loss. 
You need to handle the forward pass, compute the loss gradients, and update the weights and biases using stochastic gradient descent. 
Ensure the function processes input data as grayscale images and one-hot encoded labels, and returns the trained weights and biases for the convolutional and dense layers.

import numpy as np; np.random.seed(42); X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]); y = np.array([[1, 0]]); print(train_simple_cnn_with_backprop(X, y, 1, 0.01, 3, 1))
'''

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):

    n_samples, ht, width = X.shape
    num_classes = y.shape[1]

    weights = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    biases = np.zeroes(num_filters)
    output_height = ht - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    weights_dense = np.random.randn(flattened_size, num_classes) * 0.01
    bias_dense = np.zeroes(num_classes)
    for epoch in range(epochs):
        # batch size = 1 for SGD
        for i in range(n_samples): 
            z = np.zeroes((output_height, output_width, num_filters))
            for k in range(num_filters):
                for p in range(output_height):
                    for q in range(output_width):
                        z[p,q,k] = np.sum(X[i, p : p + kernel_size, q : q + kernel_size] * weights[:, :, k]) + biases[k]
        activation = np.max(z, 0)       
        op1 = activation.flatten()
        
        # second layer (dense layer with softmax output)
        z2 = np.dot(op1, weights_dense) + bias_dense
        exp_z = np.exp(z2 - np.max(z2)) # softmax
        softmax = exp_z / np.sum(exp_z)

        # backprop
        dz = softmax - y[i]
        dw_dense = np.outer(op1, dz)
        db_dense = dz
        dop1 = np.dot(dz, weights_dense.T)
        # reshape and backprop via relu
        dactivation = dop1.reshape(activation.shape)
        dz_conv = dactivation * (activation > 0).astype(float)

        # conv layer gradients
        dw_conv = np.zeros_like(weights)
        db_conv = np.zeros(num_filters)
        for k in range(num_filters):
            db_conv[k] = np.sum(dz_conv[:, :, k])
            for i in range(kernel_size):
                for j in range(kernel_size):
                    dw_conv[i, j, k] = np.sum(dz_conv[:, :, k] * X[i, i : i+output_height, j : j + output_width])

        # update 
        weights -= learning_rate * dw_conv
        biases -= learning_rate * dw_conv
        weights_dense -= learning_rate * dw_dense
        bias_dense -= learning_rate * db_dense

    return weights, biases, weights_dense, bias_dense


