# Full Transformer (Attention Is All You Need)
# Encoder: N layers of (self-attention + FFN)
# Decoder: N layers of (masked self-attention + cross-attention + FFN)
# + embeddings, positional encoding, output projection

import numpy as np
from pe import pos_encoding
from embeddings import embed_tokens, output_projection, init_embedding
from encoder_block import encoder_block, init_encoder_block
from decoder_block import decoder_block, init_decoder_block


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def init_transformer(vocab_size: int, d_model: int, d_ff: int, n_layers: int):
    return {
        "embedding": init_embedding(vocab_size, d_model),
        "encoder_layers": [init_encoder_block(d_model, d_ff) for _ in range(n_layers)],
        "decoder_layers": [init_decoder_block(d_model, d_ff) for _ in range(n_layers)],
    }


def encode(src_tokens: np.ndarray, params: dict, n_heads: int, d_model: int) -> np.ndarray:
    x = embed_tokens(src_tokens, params["embedding"])
    pe = pos_encoding(len(src_tokens), d_model)
    x = x + pe.squeeze(0)

    for layer_params in params["encoder_layers"]:
        x = encoder_block(x, layer_params, n_heads)
    return x


def decode(tgt_tokens: np.ndarray, enc_output: np.ndarray, params: dict, n_heads: int, d_model: int) -> np.ndarray:
    x = embed_tokens(tgt_tokens, params["embedding"])
    pe = pos_encoding(len(tgt_tokens), d_model)
    x = x + pe.squeeze(0)

    for layer_params in params["decoder_layers"]:
        x = decoder_block(x, enc_output, layer_params, n_heads)
    return x


def transformer_forward(src_tokens: np.ndarray, tgt_tokens: np.ndarray, params: dict,
                         n_heads: int, d_model: int) -> np.ndarray:
    enc_output = encode(src_tokens, params, n_heads, d_model)
    dec_output = decode(tgt_tokens, enc_output, params, n_heads, d_model)
    logits = output_projection(dec_output, params["embedding"])
    probs = softmax(logits)
    return probs


def main():
    np.random.seed(42)
    vocab_size, d_model, d_ff = 50, 16, 64
    n_heads, n_layers = 2, 2

    params = init_transformer(vocab_size, d_model, d_ff, n_layers)

    src = np.array([5, 12, 33, 7, 1])
    tgt = np.array([1, 8, 22, 3])

    probs = transformer_forward(src, tgt, params, n_heads, d_model)
    print(f"Source tokens: {src}")
    print(f"Target tokens: {tgt}")
    print(f"Output probabilities: {probs.shape}  (seq_len={len(tgt)}, vocab={vocab_size})")
    print(f"Predicted next tokens: {np.argmax(probs, axis=-1)}")

if __name__ == "__main__":
    main()
