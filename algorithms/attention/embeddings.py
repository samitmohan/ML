# Token Embeddings
# The paper scales embeddings by sqrt(d_model) so they don't get dwarfed by positional encodings
# Encoder input, decoder input, and output projection share the same weight matrix

import numpy as np


def embed_tokens(token_ids: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
    # token_ids: (seq_len,) integer indices
    # embedding_matrix: (vocab_size, d_model)
    d_model = embedding_matrix.shape[1]
    return embedding_matrix[token_ids] * np.sqrt(d_model)


def output_projection(hidden: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
    # Project decoder output back to vocab logits using shared embedding weights
    # hidden: (seq_len, d_model), output: (seq_len, vocab_size)
    logits = hidden @ embedding_matrix.T
    return logits


def init_embedding(vocab_size: int, d_model: int) -> np.ndarray:
    return np.random.randn(vocab_size, d_model) * 0.01


def main():
    np.random.seed(42)
    vocab_size, d_model, seq_len = 100, 8, 5

    E = init_embedding(vocab_size, d_model)
    tokens = np.array([2, 15, 42, 7, 99])

    embedded = embed_tokens(tokens, E)
    print(f"Token IDs: {tokens}")
    print(f"Embedded shape: {embedded.shape}")
    print(f"Scaling factor (sqrt d_model): {np.sqrt(d_model):.4f}")

    logits = output_projection(embedded, E)
    print(f"Output logits shape: {logits.shape}")

if __name__ == "__main__":
    main()
