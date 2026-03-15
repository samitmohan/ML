# Decoder Block
# Each decoder layer has three sub-layers:
#   1. Masked multi-head self-attention (causal - can't see future tokens)
#   2. Multi-head cross-attention (attends to encoder output)
#   3. Position-wise feed-forward network
# All wrapped with residual connection + layer norm

import numpy as np
from mha import compute_qkv, multi_head_attention
from masked_attention import causal_mask, softmax
from cross_attention import multi_head_cross_attention
from ffn import feed_forward, init_ffn
from residual_connection import add_and_norm, init_norm_params


def masked_multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    mask = causal_mask(seq_len)

    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    heads = []
    for h in range(n_heads):
        scores = (Q[h] @ K[h].T) / np.sqrt(d_k)
        scores = scores + mask
        attn_weights = softmax(scores)
        heads.append(attn_weights @ V[h])

    return np.concatenate(heads, axis=-1)


def init_decoder_block(d_model: int, d_ff: int):
    def init_attn_weights():
        s = np.sqrt(2.0 / (2 * d_model))
        return {
            "W_q": np.random.randn(d_model, d_model) * s,
            "W_k": np.random.randn(d_model, d_model) * s,
            "W_v": np.random.randn(d_model, d_model) * s,
            "W_o": np.random.randn(d_model, d_model) * s,
        }

    W1, b1, W2, b2 = init_ffn(d_model, d_ff)
    return {
        "self_attn": init_attn_weights(),
        "cross_attn": init_attn_weights(),
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "gamma1": np.ones(d_model), "beta1": np.zeros(d_model),
        "gamma2": np.ones(d_model), "beta2": np.zeros(d_model),
        "gamma3": np.ones(d_model), "beta3": np.zeros(d_model),
    }


def decoder_block(x: np.ndarray, enc_output: np.ndarray, params: dict, n_heads: int) -> np.ndarray:
    sa = params["self_attn"]
    ca = params["cross_attn"]

    # Sub-layer 1: masked self-attention + add & norm
    Q, K, V = compute_qkv(x, sa["W_q"], sa["W_k"], sa["W_v"])
    masked_attn_out = masked_multi_head_attention(Q, K, V, n_heads)
    masked_attn_out = masked_attn_out @ sa["W_o"]
    x = add_and_norm(x, masked_attn_out, params["gamma1"], params["beta1"])

    # Sub-layer 2: cross-attention to encoder output + add & norm
    Q_cross = x @ ca["W_q"]
    K_cross = enc_output @ ca["W_k"]
    V_cross = enc_output @ ca["W_v"]
    cross_attn_out = multi_head_cross_attention(Q_cross, K_cross, V_cross, n_heads)
    cross_attn_out = cross_attn_out @ ca["W_o"]
    x = add_and_norm(x, cross_attn_out, params["gamma2"], params["beta2"])

    # Sub-layer 3: feed-forward + add & norm
    ffn_out = feed_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
    x = add_and_norm(x, ffn_out, params["gamma3"], params["beta3"])

    return x


def main():
    np.random.seed(42)
    enc_len, dec_len, d_model, d_ff, n_heads = 6, 4, 8, 32, 2

    enc_output = np.random.randn(enc_len, d_model)
    dec_input = np.random.randn(dec_len, d_model)
    params = init_decoder_block(d_model, d_ff)

    out = decoder_block(dec_input, enc_output, params, n_heads)
    print(f"Decoder block: dec={dec_input.shape}, enc={enc_output.shape} -> {out.shape}")
    print(out)

if __name__ == "__main__":
    main()
