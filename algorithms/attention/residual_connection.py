# Add & Norm: residual connection followed by layer normalization
# output = LayerNorm(x + Sublayer(x))
# Every sub-layer (attention, FFN) in the transformer is wrapped with this

import numpy as np


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


def add_and_norm(x: np.ndarray, sublayer_output: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return layer_norm(x + sublayer_output, gamma, beta)


def init_norm_params(d_model: int):
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    return gamma, beta


def main():
    np.random.seed(42)
    seq_len, d_model = 4, 8
    x = np.random.randn(seq_len, d_model)
    sublayer_out = np.random.randn(seq_len, d_model) * 0.1

    gamma, beta = init_norm_params(d_model)
    out = add_and_norm(x, sublayer_out, gamma, beta)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output mean per position: {np.mean(out, axis=-1)}")
    print(f"Output std per position: {np.std(out, axis=-1)}")

if __name__ == "__main__":
    main()
