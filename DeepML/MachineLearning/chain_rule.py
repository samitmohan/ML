# https://www.deep-ml.com/problems/214?returnTo=paths?
import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.

    Args:
        functions: List of function names (applied right to left)
                    Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative

    Returns:
        Derivative value at x

    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """

    # Forward pass: store intermediate values
    values = [x]

    for fn in reversed(functions):
        v = values[-1]
        if fn == "square":
            values.append(v ** 2)
        elif fn == "sin":
            values.append(np.sin(v))
        elif fn == "exp":
            values.append(np.exp(v))
        elif fn == "log":
            values.append(np.log(v))
        else:
            raise ValueError(f"Unknown function: {fn}")

    # Backward pass: chain rule
    grad = 1.0
    for fn, v in zip(functions, reversed(values[:-1])):
        if fn == "square":
            grad *= 2 * v
        elif fn == "sin":
            grad *= np.cos(v)
        elif fn == "exp":
            grad *= np.exp(v)
        elif fn == "log":
            grad *= 1 / v

    return grad
