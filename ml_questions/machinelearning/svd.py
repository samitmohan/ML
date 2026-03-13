# https://www.deep-ml.com/problems/12
import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A â U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # U = 2x2 orthogonal matrix (A.At = I)
    # S = length 2 numpy arr containing singular val
    # Vt tranpose of right singular vector matrix (direction of pts)

    # first thing would be to be eigenvalues

    H = A.T @ A
    evalues, V = np.linalg.eigh(H)
    singular_val = np.sqrt(evalues)
    # sort singular values
    idx = np.argsort(singular_val)[::-1]
    S = singular_val[idx]
    V = V[:, idx]

    # U=AVÎ£â1
    U = A @ V 
    U[:, 0] /= S[0]
    U[:, 1] /= S[1]
    return (U, S, V.T)