## MNIST from Scratch only using numpy. 

Without numpy:

- np.dot
```python
def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    # num(col) of matrix == len of vector
    if len(a) == 0 or len(a[0]) != len(b):
        return -1
    res = []
    for row in a:
        dot_prod = sum(row[i] * b[i] for i in range(len(b)))
        res.append(dot_prod)
    return res

```
- np.argmax
```python
def my_argmax_1d(arr):
    """
    Returns the index of the maximum value in a 1D list or tuple.
    Assumes non-empty array.
    """
    if not arr:
        raise ValueError("my_argmax_1d() arg is an empty sequence")

    max_val = arr[0]
    max_idx = 0

    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx
```

- np.zeroes_like
```python
def my_zeros_like(arr):
    """
    Returns a new list of lists (matrix) filled with zeros,
    matching the shape of the input list of lists.
    Assumes rectangular input (all inner lists have the same length).
    """
    if not arr:
        return [] 

    if not isinstance(arr[0], list): # If the first element is not a list, assume 1D
        return [0] * len(arr)

    rows = len(arr)
    if rows == 0:
        return []
    cols = len(arr[0])
    if cols == 0:
        return [[] for _ in range(rows)] # Handle empty inner lists

    new_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    return new_matrix
```

 remaining np.random can be replaced with just random module.

```python
python3 run.py # to run
```