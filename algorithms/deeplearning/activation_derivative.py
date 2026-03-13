# https://www.deep-ml.com/problems/217?returnTo=paths
import math

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    sigmoid = 1 / (1 + math.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    tanh = math.tanh(x)
    tanh_derivative = 1 - tanh ** 2

    relu_derivative = 1.0 if x > 0 else 0.0

    return {
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'relu': relu_derivative
    }
