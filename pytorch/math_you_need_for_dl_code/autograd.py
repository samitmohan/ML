# How autograd works

import torch

weight = torch.randn(3, requires_grad=True) # 3 features, save this gradient
x = torch.tensor([1.,2.,3.])
y = (weight * x).sum() # sum of dot products 
y.backward() # gradients  are calculated here using chain rule + partial derivatives
# dy/dx = dy/dw * dw/dx (chain rule) = sum(x) -> 1 + 2 + 3 and tensor would be [1, 2, 3]

print("grad:", weight.grad)  # derivative dy/dw == x
'''
Running through the output
Verified:

grad: tensor([1., 2., 3.])

So we don't have to do this manually, torch does this calculus magic for us by calling .backward()

'''