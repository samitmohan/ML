## what

- Creating tiny auto grad which handles scalar values (like pytorch)
- Adding relu, tanh, sigmoid activation functions.
- Works for small data set (no batch normalisation, yet :P)
- Attempt to re-create PyTorch but tiny without looking it up.
- no GPU / cuda : runs on CPU

```python
from tinygrad.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

## TODO

- Figure out how to show data / memory / flops being used for neural network in kernel = .memory_usage()

inspired by https://github.com/karpathy/
