# numpy from scratch ?

Always thought of numpy as an essential 4 ml, time to make one & see how it works. Also I have nothing to do this weekend..


# why i like numpy
numpy.eye()



```python
import numpy as np

ones = np.ones(dtype=int, 
               shape=(2,2)) + 99 # [[100,100][100,100]]
print(ones) 

# zeroes also works like that, np.empty = garbage values then sets all values to =.

print(np.eye(3)) # eye is for diagonal ararys 1 
print(np.diag((1,2,3,4))) # diagonal elements 

print(ary.sum(axis=1)) # also does the same but little faster
# Can we do multiplication like this? and then add, but I mean np.dot exists for a reason.
```

## Numpy Broadcasting is pretty cool
```python
# perform vectorized ops b/w two arrays even if dimension mismatch
print("Broadcasting,adding ary (3 columns)")
ary = np.array([1,2,3])
ary2d=[[4,5,6], [7,8,9]]
print(f"Broadcasted sum: {ary2d + ary}") # shape mismatch but can still add by broadcast
```
![Broadcasting](np_broadcasting.png)


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