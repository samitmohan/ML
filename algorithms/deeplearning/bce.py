# https://www.deep-ml.com/problems/263
import numpy as np

def binary_cross_entropy( y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clamp predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return np.mean(loss)
