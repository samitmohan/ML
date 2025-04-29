# https://www.deep-ml.com/problems/3
'''
Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list [ ]
Example:
Input:

a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)

Output:

[[1, 2], [3, 4], [5, 6], [7, 8]]

Reasoning:
The given matrix is reshaped from 2x4 to 4x2.
2D → 1D flattening
Mapping indices across reshapes
How reshaping works under the hood (just like numpy)
To handle invalid reshape -> check if total elements in original 1D array = total elements in reshape 1D array

'''
import numpy as np

# reshape_matrix = 4*2
# orig_matrix = 2*4
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    orig_total = sum(len(row) for row in a) # total elements in orig
    reshape_total = new_shape[0] * new_shape[1] # total elements in reshaped
    if orig_total != reshape_total: return [] # returns empty list
    reshape_matrix = [[0] * new_shape[1] for _ in range(new_shape[0])]
    num_cols = len(a[0])
    new_rows, new_cols = new_shape # unpacking tuple
    for row in range(new_rows): # 4
        for col in range(new_cols): # 2
            flat_index = row * new_cols + col #flat_index = 0 * 2 + 0 = 0 || 0 * 2 + 1 = 1 || 1 * 2 + 0 = 2 || 1 * 2 + 1 = 3 || 2 * 2 + 0 = 4 and so on [0,1,2,3,4]
            # now map flat_index to og array
            orig_row = flat_index // num_cols # 0 // 4 = 0, 1 // 4  = 0, 2 // 4 = 0, 3 // 4 = 0, 4//4 = 1
            orig_col = flat_index % num_cols # 0 % 4 = 0, 1 % 4 = 1 , 2 % 4 = 2, 3 % 4 = 3, 4%4 = 0
            # [0,0], [0,1], [0,2], [0,3], [1][0]
            reshape_matrix[row][col] = a[orig_row][orig_col] 
    return np.array(reshape_matrix).tolist()

# This is such a shitty solution, a better way to think is to flatten the original matrix and then put those values in reshape_matrix acc to it's row, col (given in new_shape)

def reshape_matrix_better(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    # check dimensions
    row, col = len(a), len(a[0])
    new_rows, new_cols = new_shape
    if row*col != new_rows*new_cols: return []

    # flatten list
    flatten_list = [a[r][c] for r in range(row) for c in range(col)]

    # populate new matrix
    reshape_matrix = []
    index = 0
    for nr in range(new_rows):
        new_row = []
        for nc in range(new_cols):
            new_row.append(flatten_list[index])
            index += 1
        reshape_matrix.append(new_row)
    return reshape_matrix


# Thank god for numpy
def reshape_matrix_numpy(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    return np.array(a).reshape(new_shape).tolist()

if __name__ == "__main__":
    # print(reshape_matrix(a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)))
    # print(reshape_matrix_better(a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)))
    # print(reshape_matrix_numpy(a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)))
