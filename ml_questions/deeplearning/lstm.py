# https://www.deep-ml.com/problems/59
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        # Biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        """
        x: input sequence of shape (T, input_size)
        h_prev: initial hidden state (hidden_size, 1)
        c_prev: initial cell state (hidden_size, 1)

        Returns:
        - hidden_states: list of hidden states for each timestep
        - h_prev: final hidden state
        - c_prev: final cell state
        """
        T = x.shape[0]
        hidden_states = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)

            # Concatenate hidden state and input
            concat = np.vstack((h_prev, x_t))

            # Gates
            f_t = self.sigmoid(self.Wf @ concat + self.bf)      # forget gate
            i_t = self.sigmoid(self.Wi @ concat + self.bi)      # input gate
            c_hat = np.tanh(self.Wc @ concat + self.bc)         # candidate cell
            o_t = self.sigmoid(self.Wo @ concat + self.bo)      # output gate

            # Cell state update
            c_prev = f_t * c_prev + i_t * c_hat

            # Hidden state update
            h_prev = o_t * np.tanh(c_prev)

            hidden_states.append(h_prev)

        return hidden_states, h_prev, c_prev
