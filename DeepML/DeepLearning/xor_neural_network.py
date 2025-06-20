# https://www.analyticsvidhya.com/blog/2024/01/xor-problem-with-neural-networks-an-explanation-for-beginners/
import numpy as np

class Network:
    def __init__(self, inp_size, hidlayer_size, output_size, activation_fn='sigmoid', epochs=1000, lr=0.01 ) -> None:
        self.inp_size = inp_size
        self.hidlayer_size = hidlayer_size
        self.output_size = output_size
        self.activation_type = activation_fn
        self.epochs = epochs
        self.lr = lr
        self.weights, self.biases = None, None

    def activation_fn(self, x):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type== 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'softmax':
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported Activation fn : {self.activation_type}")
    
    def activation_derivative(self, a): # a is the activation fn itself
        if self.activation_type == 'sigmoid':
            return a * (1 - a)
        elif self.activation_type== 'relu':
            return (a > 0).astype(float)
        elif self.activation_type == 'tanh':
            return 1 - a**2
        elif self.activation_type == 'softmax':
            a = a.reshape(-1, -1)
            return np.diagflat(a) - np.dot(a, a.T)
        else:
            raise ValueError(f"Unsupported Activation fn : {self.activation_type}")
        
    def xavier_init(self, shape):
        ''' xavier init helps by choosing weights that preserve the variance of activations and gradients through each layer'''
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))
        weights = np.random.uniform(low=-limit, high=limit, size=shape)
        return weights
    
    def train(self, x, y):
        '''
        Input x  --> [W1, b1] --> z1 --> a1 --> [W2, b2] --> z2 --> a2 (prediction)
                                 ↑                          ↑
                             backprop dz1            backprop dz2
        z1 = w1 * x + b1
		a1 = activation(z1)
		z2 = w2 * a1 + b2
		a2 = activation(z2)
        '''
        features, samples = x.shape
        w1 = self.xavier_init((self.hidlayer_size, features)) # shape
        # make a 2D array with shape (hidden_size × 1)
        b1 = np.zeros((self.hidlayer_size, 1)) 
        w2 = self.xavier_init((self.output_size, self.hidlayer_size)) # shape
        b2 = np.zeros((self.output_size, 1))

        for _ in range(self.epochs):
            # forward pass
            z1 = np.dot(w1, x) + b1
            a1 = self.activation_fn(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = self.activation_fn(z2)

            # backward pass
            # output -> hidden
            # gradient of loss wrt z2 
            # a2 - y = derivative of mse loss wrt a2
            # if loss = 1/2 (a2 -y)^2 then a2-y is the dLoss/da2
            # need to calculate derivative of loss wrt z2 : dL/dz2
            # dL/dz2 = dL/da2 * da2/dz2 since z2 is depdendent on a2
            # dz2 = dLoss/da2 * da2/dz2 (chain rule)
            dz2 = (a2 - y) * self.activation_derivative(a2)
            # next we want gradient of loss wrt w2
            # dL/dw2 = dL/dz2 * dz2/dw2 as that's how w2 is related to loss (via z2)
            # z2 = w2*a1 + b2 so dz2/dw2 = a1
            # dL/dw2 = dL/dz2(WE ALR CALCULATED) * a1
            # dz2 has shape (output_size, n_samples) and a1 had size (hidden_size, n_samples) so we flip it a1.T so now it has (n_samples, hidden_size) shape and can be multiplied
            # how much small change in w2 affects the loss
            dw2 = np.dot(dz2, a1.T) / samples # divided bcs of scaling with batch_size
            # dL/db2 = dL/dz2 * dz2/db2 = dz2 (ALR CALC) * loss derivative wrt b2
            # loss = w2*a2 + b2 so wrt b2 deriv = 1 :: hence dL/db2 = dz2 * 1 = dz2
            db2 = np.sum(dz2, axis=1, keepdims=1) / samples # sum across batch dim to get total contribution of each sample to bias gradient

            # hidden -> input
            # dz1 = dL/dz1 
            # dz2 is gradient loss output layer wrt input z2
            # gradient of loss wrt a1 : dL/da1 = dL/dz2 * dz2/da1 but dl/dz2 = dz2 and dz2/da1 = w2 bcs z2 = w2*a1
            # w2 maps from a1 to z2 so going backward requires a transpose
            dz1 = np.dot(w2.T, dz2) * self.activation_derivative(a1)
            # dL/dw1 = dL/dz1 (alr calc) * dz1/dw1 (x.T)
            dw1 = np.dot(dz1, x.T) / samples
            db1 = np.sum(dz1, axis=1, keepdims=True) / samples

            
            # update weights and biases
            w1 -= self.lr * dw1
            w2 -= self.lr * dw2
            b1 -= self.lr * db1
            b2 -= self.lr * db2
        
        # storing trained parameters
        self.weights = {
            'w1': w1,
            'w2': w2,
        }
        self.biases = {
            'b1': b1,
            'b2': b2,
        }

            
    def predict(self, x_test):
        b1 = self.biases['b1']
        b2 = self.biases['b2']
        w1 = self.weights['w1']
        w2 = self.weights['w2']
        z1 = np.dot(w1, x_test) + b1
        a1 = self.activation_fn(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = self.activation_fn(z2)
        pred = a2
        return (pred > 0.5).astype(int)

if __name__ == '__main__':
    # XOR input and output
    # 2 features (each input sample has 2 values: x1 and x2)
	# 4 samples (4 total examples)
    x = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # Shape (2, 4) : 2 features (each input sample has 2 values x1 and x2) and 4 total samples (examples)
    y = np.array([[0, 1, 1, 0]])  # Shape (1, 4) we need these 4 samples for gradient descent and 1 feature to intialise weight matrix w1 size 

    # Initialize and train the network
    net = Network(inp_size=2, hidlayer_size=4, output_size=1, activation_fn='sigmoid', epochs=20000, lr=0.1)
    net.train(x, y)

    # Prediction
    preds = net.predict(x)
    print("Predictions:", preds)
    print("Ground Truth:", y)

'''
XOR works
Predictions: [[0 1 1 0]]
Ground Truth: [[0 1 1 0]]

More notes;
w1 is shape (hidden_size, input_size)
x is shape (input_size, n_samples)
So, np.dot(w1, x) → shape (hidden_size, n_samples)
Therefore, b1 must broadcast correctly to match shape (hidden_size, n_samples)

You’re adding the same bias value for each hidden neuron across all samples in a batch:
So you want one bias per hidden neuron ⇒ hidden_size rows
And you want that same bias applied to each sample ⇒ broadcast across columns

Hence, (hidden_size, 1) makes it work seamlessly.
axis=0 → operates down the rows (i.e., across samples for each neuron).
axis=1 → operates across the columns (i.e., across all samples for each neuron).

You want the total gradient contribution from all samples for each bias unit (1 in this case), so summing across columns (samples) makes sense.
'''