import torch
import torch.nn.functional as F
from torch.autograd import grad


# print(torch.__version__)
# print(torch.cuda.is_available()) # false
# print(torch.backends.mps.is_available()) # true

# -- tensor -- 


tensor0d = torch.tensor(1) # scalar
tensor1d = torch.tensor([1,2,3]) # vector
tensor2d = torch.tensor([[1, 2], [3, 4]]) # matrix
tensor3d = torch.tensor([[1, 2], [3, 4], [5, 6]])  # nested list
# print(tensor0d.dtype) # with int its int64, with float is float32
# print(tensor1d)
# print(tensor2d)
# print(tensor3d)
# print(tensor3d.shape)
# print(tensor3d)
# tensor3d.reshape(2, 3)

# print(tensor3d.matmul(tensor3d.T)) # same as @

# -- autograd --
y = torch.tensor([1.0]) # true label
x1 = torch.tensor([1.1]) # inp
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print(loss)
grad_L_w1 = grad(loss, w1, retain_graph=True)
print(f"Gradient of Loss wrt Weight: {grad_L_w1}")
grad_L_b = grad(loss, b, retain_graph=True)
print(f"Gradient of Loss wrt Bias: {grad_L_b}")

loss.backward() # instead
print(w1.grad)
print(b.grad)
