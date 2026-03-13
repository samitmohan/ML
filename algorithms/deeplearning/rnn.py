# https://www.deep-ml.com/problems/54?from=Deep%20Learning

import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
    x_seq = np.array(input_sequence)
    h_t = np.array(initial_hidden_state)
    Wx, Wh, b = np.array(Wx), np.array(Wh), np.array(b)
    for t in range(len(x_seq)):
        x_t = x_seq[t]
		h_t = np.tanh(Wx @ x_t + Wh @ h_t + b)


	return h_t.tolist()

