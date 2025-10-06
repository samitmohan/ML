# https://sebastianraschka.com/blog/2020/numpy-intro.html

# Motivation
'''
- Dot Product
First in Python
'''
import time
import numpy as np

def python_dot(w, x):
    z = 0
    for i in range(len(w)):
        z += w[i] * x[i]
    return z

def numpy_dot(w, x):
    return np.dot(w, x)


def main():
    vector_size = 1_000_000
    # python
    wlist = list(range(vector_size))
    xlist = list(range(vector_size))
    start = time.time()
    res = python_dot(wlist, xlist)
    end = time.time()
    print(res)
    print(f" Time for python vector: {(end-start) * 1000:.4f} ms ")



    # num-py
    w_np = np.arange(vector_size)
    x_np = np.arange(vector_size)
    numpy_dot(w_np, x_np)
    start = time.time()
    res = numpy_dot(w_np, x_np)
    end = time.time()
    print(res)
    print(f" Time for numpy vector: {(end-start) * 1000:.4f} ms ") # over 100x faster


if __name__ == "__main__":
    main()
