# In order to reshape arrays so that they can compute matrix multiplications, numpy provides .reshape()
import numpy as np

m = np.array([1,2,3]) # m = 3 
n = np.array([1,2]) # n = 2
print(f"n : {n}")
print(f"m's shape : {m.shape}, n's shape : {n.shape}")
# cant matrix multiply them rn, different shapes -> reshape so that (m * p) and (p * n) = p = p (matmul rules)
m1 = m.reshape((3,1))
n1 = n.reshape((1,2))
print("\n")
print(f"m1 now: {m1}\n, n1 now: {n1}")
print(np.dot(m1, n1))

#try:
#    print(np.dot(m, n)) # ValueError: shapes (3,) and (2,) not aligned: 3 (dim 0) != 2 (dim 0)
#except ValueError as e:
    # right method

# make sure reshaped array has same elements as original. numpy does this (figuring out how many elem to put along an axis)
print("reshaping")
n.reshape(-1, 2) # not a copy.
print(n.shape)
# also flattening -> reshape(-1)
print(m1.reshape(-1)) # 1,2,3 instead of (3,1)[1,2,3]

# flatten does same (copy of array), and ravel (memory view like reshape)
# merging different arrays without creating new array -> concatenate
print("Concatenating")
ary = np.array([1,2,3])
ary2 = np.array([4,5,6])
print(np.concatenate([ary, ary2]))
ary = np.array([[1,2,3]])
print(np.concatenate((ary, ary), axis=1)) # axis=1, across column -> [[1,2,3,1,2,3]], axis=0, across row -> [[1,2,3], [1,2,3]]

