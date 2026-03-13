# https://www.deep-ml.com/problems/94

import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = X @ W_q # (seq_len, d_model) * (d_model, d_model)
    K = X @ W_k 
    V = X @ W_v
    return Q, K, V


def softmax(x, axis = -1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    dk = Q.shape[1] # seq_len, d_k (we want this)
    scores = (Q @ K.T) / np.sqrt(dk)
    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    d_k = d_model // n_heads

    # Split into heads
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Apply self-attention per head
    heads = []
    for h in range(n_heads):
        head_output = self_attention(Q[h], K[h], V[h])
        heads.append(head_output)

    # Concatenate heads
    output = np.concatenate(heads, axis=-1)
    return output
