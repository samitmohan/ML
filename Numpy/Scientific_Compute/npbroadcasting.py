# Broadcasting in numpy
import numpy as np

# perform vectorized ops b/w two arrays even if dimension mismatch
print("Broadcasting,adding ary (3 columns)")
ary = np.array([1,2,3])
ary2d=[[4,5,6], [7,8,9]]
print(f"Broadcasted sum: {ary2d + ary}") # shape mismatch but can still add by broadcast
