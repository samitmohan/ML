# Memory views and copies
import numpy as np

# Numpy is Memory efficient
# views is great since it avoids making unnecessary copies of array to save memory resources.
# access first row of arr, assign to variable and modify that var
ary = np.array([[1,2,3],[4,5,6]])
first_row = ary[0]
print(f"First row: {first_row}")
first_row += 100
print(f"Modified first row: {first_row}") 
print(ary) # changing value of first_row also affected the original array. (ary[0] created a view of first row in array and its elements were then incremented)
# same with slicing (also creates views)
# when we want to force a copy (like python) we can use copy
copied_first_row = ary[0].copy()
copied_first_row += 100
print(copied_first_row) # array([201, 202, 203])
print(ary) # still [[101 102 103] [  4   5   6]]

# fancy indexing
print("Fancy indexing")
ary = np.array([[1,2,3],
                [4,5,6]])
print(ary[:, [0, 2]]) # first and last column

# boolean mask also pretty cool
greater3_mask = ary > 3
print(greater3_mask, type(greater3_mask), sep="\n")

