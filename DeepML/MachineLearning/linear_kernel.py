# https://www.deep-ml.com/problems/45
# The linear kernel between x1 and x2 is computed as:1*4 + 2*5 + 3*6 = 32


import numpy as np

def kernel_function(x1, x2):
    result = np.inner(x1, x2)
    return result
