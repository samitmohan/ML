import numpy as np

def softmax(x, axis=-1):
	x = x - np.max(x, axis=axis, keepdims=True)
	exp = np.exp(x)
	return exp / np.sum(exp, axis=axis, keepdims=True)

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	dk = Q.shape[-1]
	scores = (Q @ K.T) / np.sqrt(dk)
	# apply mask
	masked_score = scores + mask
	masked_weights = softmax(masked_score, axis=-1)
	output = masked_weights @ V
	return output 


# If mask not given
def causal_mask(seq_len: int) -> np.ndarray:
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask * -1e9


def main():
    mask = causal_mask(Q.shape[0])
    output = masked_attention(Q, K, V, mask)

main()

