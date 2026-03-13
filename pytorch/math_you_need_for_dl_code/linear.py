# class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# y = x * W.transpose + b
'''
x → input vector (size = in_features)
W → weight matrix (out_features x in_features)
b → bias vector (out_features)
y → output vector (out_features)
'''
import torch

def main():
    input_data = [[1., 2., 3.],
                  [4., 5., 6.]]   # shape: (rows/batch=2, columns/features=3)
    x = torch.tensor(input_data) # convert to tensor
    formula = torch.nn.Linear(in_features=3, out_features=1, bias=True)
    y = formula(x)
    print(f"Weights : {formula.weight} and Bias : {formula.bias}")
    print(f"input shape: {x.shape}") 
    print(f"output shape: {y.shape}") 
    print(f"output of y = mx + c : {y}")
main()

'''
Running through the output

Out input is matrix:
[
[1, 2, 3],
[4, 5, 6]
]  
Shape: (2 * 3) 2 rows and 3 columns

Our random weights (the W in y = WX + B and Bias) shape will be (1 * 3) -> (out_features, in_features)
Weights: tensor([[0.4468, 0.0444, 0.4144]], requires_grad=True) and 
Bias : tensor([0.3921], requires_grad=True)

Which makes sense since our in_features = 3 (weights) and bias = 1
So shape of our random W matrix = (1x3) and shape of B = (1,) # scalar

Now we need to do y = WX + bias (Slope equation) -> also known as torch.nn.Linear
Let's write it in shape form
y = (2 * 3) * (1 * 3) + bias (1,)

Can't matrix multiply like this, since (p*q) * (q*r) = p * r for matrix multiply, hence we need to transpose our Weight matrix
W.transpose !! so now shape of W becomes (3 * 1) and y = (2 * 3) * (3 * 1) + bias(1,) which can now be done

Another question you might have -> once we get X * W.Transpose of (2*1) how are we adding bias (1,) to it? Since bias is a scalar and not of the same shape how are 2 matrices of different shapes being added?
PyTorch (you beauty) handles this internally by broadcasting / changing the bias from scalar (1, ) to (2, 1) {2 rows, 1 column both have same values of the original bias 0.3921} -> Now you can easily add the X * W.T + B because both all have same shapes.

So after matrix multiply output shape will be (2 * 1), we can verify this by seeing output shape of y:
input shape: torch.Size([2, 3])
weight shape after transpose: torch.Size([3, 1])
output shape: torch.Size([2, 1]) which means 2 rows 1 column

What do these 2 rows and 1 column mean -> the gradients of weight and bias respectively.

nn.Linear does this whole thing (including transpose, the whole backpropagation thing (since requires_grad=True)) and stores it.

When we output y -> we get 2 gradients (one is weight and one is for bias) which tells us how much we need to move the weights and bias by to move in the correct direction of loss (to minimise it)

output of y = mx + c : tensor([[2.1708], [4.8874]], grad_fn=<AddmmBackward0>)

so we need to move -2.17 for our weights and -4.8 for our bias (why -? because we move in the negative direction of the gradients to find the minimum of the loss function)
'''

# formula can also be written as Linear class
'''
This is what happens under the hood for Linear.

class Linear:
    def __init__(self):
        self.W = random()
        self.b = random()

    def forward(self, x):
        return x @ self.W.T + self.b

'''

'''
Why this matters for Transformers / LLMs

Everywhere you see: 
- W_q, W_k, W_v
- W_o
- FFN/MLP layers

They are just nn.Linear layers with different shapes.
Example:
W_q = nn.Linear(d_model, d_model)
Same concept. Bigger matrices.
'''