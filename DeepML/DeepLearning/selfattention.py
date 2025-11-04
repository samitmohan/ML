#https://www.deep-ml.com/problems/53
import numpy as np

def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = Q.shape[-1] # column of query : scaling val
    attn_logits = Q @ K.T # Q : [3,2], K : [3,2] can't do matmul hence t
    attn_logits = attn_logits/np.sqrt(d_k)
    attn_logits = softmax(attn_logits)
    attention_output = attn_logits @ V


	return attention_output


