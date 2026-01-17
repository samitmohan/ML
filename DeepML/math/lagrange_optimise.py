# https://www.deep-ml.com/problems/314?returnTo=paths

import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)

    # Build KKT system
    KKT = np.block([
        [Q, a.reshape(-1, 1)],
        [a.reshape(1, -1), np.zeros((1, 1))]
    ])

    rhs = np.concatenate([-c, [b]])

    # Solve KKT system
    solution = np.linalg.solve(KKT, rhs)

    x = solution[:2]
    lam = solution[2]

    # Compute objective value
    objective = 0.5 * x.T @ Q @ x + c.T @ x

    return {
        'x': np.round(x, 4).tolist(),
        'lambda': round(lam, 4),
        'objective': round(objective, 4)
    }
