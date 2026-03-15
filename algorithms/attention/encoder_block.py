# Encoder Block
# Each encoder layer has two sub-layers:
#   1. Multi-head self-attention
#   2. Position-wise feed-forward network
# Both wrapped with residual connection + layer norm

import numpy as np
from mha import compute_qkv, multi_head_attention
from ffn import feed_forward, init_ffn
from residual_connection import add_and_norm, init_norm_params


def init_encoder_block(d_model: int, d_ff: int):
    W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (2 * d_model))
    W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (2 * d_model))
    W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (2 * d_model))
    W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (2 * d_model))
    W1, b1, W2, b2 = init_ffn(d_model, d_ff)
    gamma1, beta1 = init_norm_params(d_model)
    gamma2, beta2 = init_norm_params(d_model)
    return {
        "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "gamma1": gamma1, "beta1": beta1,
        "gamma2": gamma2, "beta2": beta2,
    }


def encoder_block(x: np.ndarray, params: dict, n_heads: int) -> np.ndarray:
    # Sub-layer 1: multi-head self-attention + add & norm
    Q, K, V = compute_qkv(x, params["W_q"], params["W_k"], params["W_v"])
    attn_out = multi_head_attention(Q, K, V, n_heads)
    attn_out = attn_out @ params["W_o"]  # output projection
    x = add_and_norm(x, attn_out, params["gamma1"], params["beta1"])

    # Sub-layer 2: feed-forward + add & norm
    ffn_out = feed_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
    x = add_and_norm(x, ffn_out, params["gamma2"], params["beta2"])

    return x


def main():
    np.random.seed(42)
    seq_len, d_model, d_ff, n_heads = 4, 8, 32, 2

    x = np.random.randn(seq_len, d_model)
    params = init_encoder_block(d_model, d_ff)
    out = encoder_block(x, params, n_heads)
    print(f"Encoder block: {x.shape} -> {out.shape}")
    print(out)

if __name__ == "__main__":
    main()
