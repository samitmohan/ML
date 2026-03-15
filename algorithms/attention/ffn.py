# Position-wise Feed-Forward Network
# FFN(x) = max(0, xW1 + b1)W2 + b2
# Two linear transforms with a ReLU in between, applied identically to each position

import numpy as np


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    # x: (seq_len, d_model), W1: (d_model, d_ff), W2: (d_ff, d_model)
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU activation
    output = hidden @ W2 + b2
    return output


def init_ffn(d_model: int, d_ff: int):
    # Xavier initialization
    W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def main():
    np.random.seed(42)
    seq_len, d_model, d_ff = 4, 8, 32
    x = np.random.randn(seq_len, d_model)
    W1, b1, W2, b2 = init_ffn(d_model, d_ff)
    out = feed_forward(x, W1, b1, W2, b2)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(out)

if __name__ == "__main__":
    main()
