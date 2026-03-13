import numpy as np
import pytest

from algorithms.attention.self_attention import (
    softmax as sa_softmax,
    compute_qkv as sa_compute_qkv,
    self_attention as sa_self_attention,
)
from algorithms.attention.mha import (
    compute_qkv,
    self_attention,
    multi_head_attention,
)
from algorithms.attention.pe import pos_encoding


class TestSelfAttention:
    def test_output_shape(self):
        seq_len, d_model = 4, 8
        X = np.random.randn(seq_len, d_model)
        W_q = np.random.randn(d_model, d_model)
        W_k = np.random.randn(d_model, d_model)
        W_v = np.random.randn(d_model, d_model)
        Q, K, V = compute_qkv(X, W_q, W_k, W_v)
        output = self_attention(Q, K, V)
        assert output.shape == (seq_len, d_model)

    def test_single_token(self):
        """With one token, attention output should equal V."""
        d = 4
        X = np.random.randn(1, d)
        W = np.eye(d)
        Q, K, V = compute_qkv(X, W, W, W)
        output = self_attention(Q, K, V)
        np.testing.assert_allclose(output, V, atol=1e-6)


class TestMultiHeadAttention:
    def test_output_shape(self):
        seq_len, d_model, n_heads = 4, 8, 2
        X = np.random.randn(seq_len, d_model)
        W = np.random.randn(d_model, d_model)
        Q, K, V = compute_qkv(X, W, W, W)
        output = multi_head_attention(Q, K, V, n_heads)
        assert output.shape == (seq_len, d_model)

    def test_single_head_matches_self_attention(self):
        """With 1 head, MHA should match regular self-attention."""
        seq_len, d_model = 3, 4
        np.random.seed(42)
        X = np.random.randn(seq_len, d_model)
        W = np.random.randn(d_model, d_model)
        Q, K, V = compute_qkv(X, W, W, W)
        sa_out = self_attention(Q, K, V)
        mha_out = multi_head_attention(Q, K, V, n_heads=1)
        np.testing.assert_allclose(mha_out, sa_out, atol=1e-6)


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(5)
        result = sa_softmax(x)
        assert abs(np.sum(result) - 1.0) < 1e-6

    def test_all_positive(self):
        x = np.array([-100, 0, 100])
        result = sa_softmax(x)
        assert np.all(result > 0)

    def test_numerical_stability(self):
        x = np.array([1000, 1001, 1002])
        result = sa_softmax(x)
        assert not np.any(np.isnan(result))
        assert abs(np.sum(result) - 1.0) < 1e-6


class TestPositionalEncoding:
    def test_output_shape(self):
        position, d_model = 10, 16
        pe = pos_encoding(position, d_model)
        assert pe.shape == (1, position, d_model)

    def test_invalid_input(self):
        assert pos_encoding(0, 16) == -1
        assert pos_encoding(5, 0) == -1

    def test_sin_cos_pattern(self):
        """Even indices use sin, odd indices use cos."""
        pe = pos_encoding(2, 4)
        # At position 0, sin(0) = 0 for even dims
        assert abs(pe[0, 0, 0]) < 1e-3  # sin(0) ~ 0
        assert abs(pe[0, 0, 1] - 1.0) < 1e-2  # cos(0) ~ 1
