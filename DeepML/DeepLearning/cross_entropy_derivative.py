# https://www.deep-ml.com/problems/220?returnTo=paths
import math

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)
        
    Returns:
        Gradient vector where gradient[i] = dL/d(logits[i])
    """
    # Numerically stable softmax
    max_logit = max(logits)
    exp_logits = [math.exp(z - max_logit) for z in logits]
    sum_exp = sum(exp_logits)
    softmax = [e / sum_exp for e in exp_logits]

    # Gradient: softmax - one_hot(target)
    grad = softmax[:]
    grad[target] -= 1.0

    return grad

# First compute softmax: p = [0.09, 0.2447, 0.6652]. The one-hot target vector is y = [1, 0, 0]. The gradient is simply p - y = [0.09 - 1, 0.2447 - 0, 0.6652 - 0] = [-0.91, 0.2447, 0.6652]. The negative gradient for class 0 indicates we should increase that logit to reduce loss.