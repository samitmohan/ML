import numpy as np

def cross_product(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    # Cross product is only defined for 3D vectors
    if a.shape != (3,) or b.shape != (3,):
        return -1

    return np.cross(a, b)
