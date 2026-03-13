# always useful in deep learning. 
# initial values of model param.
import numpy as np
np.random.seed(123) # always going to be consistent while producing random numbers ("results are reproducible")
print(np.random.randn(4,))
# another way to do this
range2 = np.random.RandomState(123)
print(range2.rand(3))
