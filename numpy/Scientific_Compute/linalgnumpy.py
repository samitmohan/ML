# Time for my favourite topic

import numpy as np
row_vector = np.array([1,2,3])
print(row_vector)
# column_vector = np.array([[1], [2], [3]])
column_vector = np.array([1,2,3]).reshape(-1, 1) # only 1 column, can figure out rows by yourself (numpy beautiful)
print(column_vector)
# instead of reshaping 1d array -> 2d array we can simply add new axis
row_vector[:, np.newaxis] # now it behaves like column vector

# To perform matrix multiplication between matrices, we learned that number of columns of the left matrix must match the number of rows of the matrix to the right. In NumPy, we can perform matrix multiplication via the matmul function:

matrix = np.array([[1,2,3], 
                  [4,5,6]])
print(f"Mat Mul : {np.matmul(matrix, column_vector)}") # (2,3) * (3,1) = (2,1) = [14,34]
# dot also does the same thing
print(np.dot(matrix, row_vector))
# @ also -> matrix @ row_vector
# transpose
print("\n")
print(f"Original: {matrix}")
print(f"Transpose: {matrix.transpose()}") # or matrix.T


