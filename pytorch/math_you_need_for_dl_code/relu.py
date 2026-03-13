'''
Docstring for math_you_need_for_dl_code.relu

Simplest non-linearity.
Why do we even need non-linearity because if we just pass
f(x) to next layer it'll be g(f(x)) and then to next layer it will be z(g(f(x)))
This is because passing linear functions makes it just composite linear function (another linear function) and we can't extract any useful data out of it.
Linear → Linear → Linear collapses into One big Linear



torch.nn.ReLU: Kill negative values. Let positive values pass

 RELU breaks linearity so networks can:
 -model complex functions
 -learn hierarchies
- represent interactions

ReLU(x)=max(0,x)
No parameters, no learning, just adds non-linearity. 

'''

import torch

def main():
    x = torch.tensor([[-2.0, -0.5, 0.0, 1.5, 3.0]])

    relu = torch.nn.ReLU()

    y = relu(x)

    print(y) # tensor([[0.0000, 0.0000, 0.0000, 1.5000, 3.0000]])

main()

'''
Running through the output
Input: any shape
Output: same shape
Internally:
if x < 0:
    x = 0

If we backprop, d/dx(RELU(x)) = 1 if x > 0 else 0 (positive neurons -> learn normally, negative neurons -> zero gradient)

In LLMs a variant of Relu called GELU() is used which is smoother, nonzero gradient everywhere and better for languagte.

You can look up graphs of RELU and GELU and instantly tell why it is better.
'''
