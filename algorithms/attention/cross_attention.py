# Cross-Attention: decoder attends to encoder output
# Q comes from decoder, K and V come from encoder
# This is how the decoder "reads" what the encoder processed

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cross_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Q: (dec_seq_len, d_k) from decoder
    # K, V: (enc_seq_len, d_k) from encoder
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)  # (dec_seq_len, enc_seq_len)
    attn_weights = softmax(scores)
    output = attn_weights @ V  # (dec_seq_len, d_k)
    return output


def multi_head_cross_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    dec_len, d_model = Q.shape
    enc_len = K.shape[0]
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    Q = Q.reshape(dec_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(enc_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(enc_len, n_heads, d_k).transpose(1, 0, 2)

    heads = []
    for h in range(n_heads):
        heads.append(cross_attention(Q[h], K[h], V[h]))

    return np.concatenate(heads, axis=-1)


def main():
    np.random.seed(42)
    enc_len, dec_len, d_model, n_heads = 6, 4, 8, 2

    enc_out = np.random.randn(enc_len, d_model)
    dec_hidden = np.random.randn(dec_len, d_model)

    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1

    Q = dec_hidden @ W_q
    K = enc_out @ W_k
    V = enc_out @ W_v

    out = multi_head_cross_attention(Q, K, V, n_heads)
    print(f"Decoder queries: {dec_hidden.shape}, Encoder context: {enc_out.shape}")
    print(f"Cross-attention output: {out.shape}")
    print(out)

if __name__ == "__main__":
    main()
