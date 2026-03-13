# ND arrays in numpy
import time

import numpy as np
a = [1.,2.,3.]
print(type(np.array(a))) #ndarray
print(type(a))
lst = [[1,2,3], [4,5,6]]
ar2d= np.array(lst)
print(ar2d, type(ar2d), sep="\nType= ")
print("Data Type: ", ar2d.dtype)
print("Shape: ", ar2d.shape)
print("Dimensions: ", ar2d.ndim)
print("Size : ", ar2d.size)

# axis = 0 (1st dimension) = rows, axis = 1 = columns (2nd dimension)

int32_ar2d = ar2d.astype(np.int32)
print(int32_ar2d)
print(int32_ar2d.dtype)

# some cool stuff you can do with numpy

ones = np.ones(dtype=int, 
               shape=(2,2)) + 99 # [[100,100][100,100]]
print(ones) 

# zeroes also works like that, np.empty = garbage values then sets all values to =.

print(np.eye(3)) # eye is for diagonal ararys 1 
print(np.diag((1,2,3,4))) # diagonal elements 

# arange: range in python (takes interval) linspace
print("NP ARANGE")
print(np.arange(4,10)) # or np.arange(5) for 0,1,2,3,4, can use step also:: np.arange(1,11,0.1)

# The linspace function is especially useful if we want to create a particular number of evenly spaced values in a specified half-open interval:
print("Linspace") 
print(np.linspace(6,15, num=50)) # generates num amount of samples between start & stop

# indexing
print("Array indexing")



ary = np.array([[1, 2, 3],
                [4, 5, 6]]) # -1, -1 for lower right

print(ary[0, -2]) # first row, second last element
print(ary[1, 2]) # last
print(ary[:, 0]) # first column
print(ary[0,:]) # first row

# Array math
print("Array math")

lst = [[1,2,3], [4,5,6]]
for row_idx, row_val in enumerate(lst):
    for col_idx, col_val in enumerate(row_val):
        lst[row_idx][col_idx] += 1
print(lst)

# using lst comprehensions
lst = [[cell + 1 for cell in row] for row in lst]
print(lst)

# using numpy ufunc : element wise scalar addition
ary = np.array([[1,2,3], 
                [4,5,6]])
print(np.add(ary, 1))
print("Square Rooting")
print(np.sqrt(ary))

# sum along a given axis
start_time = time.perf_counter_ns()
print(np.add.reduce(ary, axis=1)) # 1+2+3 or axis=0 : 1+4, 2+5, 3+6
end_time = time.perf_counter_ns()
print(f"Time taken for reduce: {end_time - start_time} ns")

start_time = time.perf_counter_ns()
print(ary.sum(axis=1)) # also does the same but little faster
end_time = time.perf_counter_ns()
print(f"Time taken for sum: {end_time - start_time} ns")

# Can also add like this
print(ary)
print(type(ary))
print(ary + 1) # adds one to every element, clean or print(np.add(ary, 1))




