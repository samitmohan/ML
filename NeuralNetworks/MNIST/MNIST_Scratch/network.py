import numpy as np
# MNIST basics
"""
if we want to create a Network object with 2 neurons in the first layer, 3 neurons in the second layer, and 1 neuron in the final layer:
    net = Network([2, 3, 1])

net.weights[1] is a Numpy matrix storing the weights connecting the second and third layers of neurons (Python counts from 0)
"""
class Network:
    def __init__(self, sizes) -> None:
        ''' This creates a network with layers and indicates how many neurons in each layer (sizes)'''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # randomly initialise the weights and bias
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z) -> float:
        ''' Activation function to convert probability in range 0 -> 1 '''
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z) -> float:
        ''' Derivative of sigmoid : we need this for calculating partial derivative of cost function '''
        return self.sigmoid(z) * (1 - self.sigmoid(z)) # basic calculus

    def forward(self, activation) -> float:
        ''' Return output of neural network '''
        for b, w in zip(self.bias, self.weights):
            activation = self.sigmoid(np.dot(w, activation) + b) # sigmoid(wx + b) (wx = dot product if you think about how matrix works)
        return activation

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.  
        The "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs.
        If "test_data" is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.
        epochs is just a fancy name for number of iterations.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data) # otherwise
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # for each mini_batch we apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                accuracy = self.evaluate(test_data)
                accuracy_percentage = (accuracy / n_test) * 100
                print( f"Epoch {epoch}: {accuracy} / {n_test}") 
            else:
                print(f"Epoch {epoch} complete") 

        print(f"Accuracy : {accuracy_percentage:.2f}%")
            
    def update_mini_batch(self, mini_batch, learning_rate):
        ''' Update weights and biases by applying Gradient Descent to single mini batch (list of tuples (x,y) representing training inputs and their desired outputs for this mini batch) '''
        # initialise gradients to 0
        # the sum of the gradients of the loss function with respect to the biases for all examples in the mini-batch.
        sum_of_bias_gradients = [np.zeros_like(b) for b in self.bias]
        sum_of_weight_gradients = [np.zeros_like(w) for w in self.weights]
        # Calculate gradient for each example in mini batch.
        for x, y in mini_batch:
            grad_b, grad_w = self.backprop(x, y)
            # update sum of bias and weight gradients (elementwise addition for gradients)
            # the overall gradient used to update the weights and biases is the average of the gradients computed for each individual training example within that mini-batch.
            sum_of_bias_gradients = [bias_sum + gradient_bias for bias_sum, gradient_bias in zip(sum_of_bias_gradients, grad_b)]
            sum_of_weight_gradients = [weight_sum + gradient_weight for weight_sum, gradient_weight in zip(sum_of_weight_gradients, grad_w)]
        
        mini_batch_size = len(mini_batch)
        learning_rate_scaled = learning_rate / mini_batch_size
        # update weights and biases using avg gradients
        self.weights = [
            w - learning_rate_scaled * weight_sum
            for w, weight_sum in zip(self.weights, sum_of_weight_gradients)
        ]

        self.biases = [
            b - learning_rate_scaled * bias_sum
            for b, bias_sum in zip(self.bias, sum_of_bias_gradients)
        ]

    def backprop(self, x, y):
        """
        Calculates the gradient for the cost function C_x
        for a single training example (x, y) using backpropagation.

        Args:
            x (np.ndarray): The input feature vector.
            y (np.ndarray): The true label/target output vector.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing
            (grad_b, grad_w), which are layer-by-layer lists of numpy
            arrays representing the gradients for biases and weights,
            respectively.
        """
        grad_b, grad_w = [np.zeros_like(b.shape) for b in self.bias], [np.zeros(w.shape) for w in self.weights]
        # forward pass
        # input layer activation and list to store all activations (layer by layer)
        activation = x
        activations = [x]
        all_z = [] # list to store all zs = (wx + b) and activation is basically sigmoid(z)
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            all_z.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backward pass
        # activations[-1] is the output layer's activation (by output layer I mean last layer)
        # all_z[-1] is the weighted input for the output layer
        error = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(all_z[-1])
        # update last layers
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activations[-2].T) # a^(L-2) * error = partial derivative of cost wrt weight of last layer

        # Iterate backward through the layers (from second-to-last to input)
        # We iterate in reverse order of layers (from output towards input).
        # Layer 1 is the first hidden layer (weights[0], biases[0]).
        # Layer L-1 is the last hidden layer (weights[-2], biases[-2]).
        for layer_idx in reversed(range(self.num_layers - 1)):
            # skip last layer (we handled that above)
            if layer_idx == self.num_layers - 2: 
                continue 
            z = all_z[layer_idx]
            sd = self.sigmoid_derivative(z)
            # Error propagation: delta for current layer from delta of next layer
            error = np.dot(self.weights[layer_idx + 1].T, error) * sd

            # Gradients for current layer's biases and weights
            grad_b[layer_idx] = error
            grad_w[layer_idx] = np.dot(error, activations[layer_idx].T)

        return grad_b, grad_w

    
    def cost_derivative(self, output_activations, y):
        ''' Return prediction - actual '''
        return output_activations - y


    def evaluate(self, test_data):
        ''' Return the number of test inputs for which the network outputs correct result '''
        test_results = [
                        (np.argmax(self.forward(x)), y) for (x, y) in test_data
                        ]
        return sum(int(x == y) for (x, y) in test_results)