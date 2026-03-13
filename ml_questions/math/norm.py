# https://www.deep-ml.com/problems/328


import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    if norm_type == "l1":
        # Sum of absolute values
        return float(np.sum(np.abs(arr)))
    
    elif norm_type == "l2":
        # Euclidean norm
        return float(np.sqrt(np.sum(arr ** 2)))
    
    elif norm_type == "frobenius":
        # Frobenius norm (same as L2 but explicitly for matrices)
        return float(np.sqrt(np.sum(arr ** 2)))
    
    else:
        raise ValueError("norm_type must be one of: 'l1', 'l2', or 'frobenius'")
