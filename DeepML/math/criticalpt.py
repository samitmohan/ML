# https://www.deep-ml.com/problems/311?returnTo=paths
import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10) -> str:
    """
    Classify a critical point based on its Hessian matrix.
    
    Args:
        hessian: Square Hessian matrix (n x n) as numpy array
        tol: Tolerance for considering eigenvalues as zero
        
    Returns:
        Classification string: 'local_minimum', 'local_maximum', 
        'saddle_point', or 'inconclusive'
    """
    eigenval = np.linalg.eigvals(hessian)
    if np.any(np.abs(eigenval) <= tol): return "inconclusive"
    # all pos -> local minima
    if np.all(eigenval>0): return "local_minimum"
    # all neg -> local maxima
    if np.all(eigenval<0): return "local_maximum"
    # mixed signs -> saddlept
    return 'saddle_point'
