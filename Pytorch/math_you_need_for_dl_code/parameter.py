'''
Docstring for math_you_need_for_dl_code.parameter

It is how you tell PyTorch: “this tensor should be learned.”
nn.Parameter(data, requires_grad=True)

It is a thin wrapper around torch.Tensor.

What changes:
- It becomes part of the models parameters
- Optimizers will update it
- It appears in model.parameters()

Why this?

Let's say
self.weight = torch.randn(10, 10)
- PyTorch ignores it during training.
But if you write:
self.weight = nn.Parameter(torch.randn(10, 10))
- PyTorch learns it

As simple as that.
'''
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale

model = MyModule()
print(list(model.parameters()))

'''
Running through the output:
[Parameter containing: tensor(1., requires_grad=True)]

Usually in LLMs we use nn.Linear (makes our lives faster) but if you want to manually describe the parameters you can use:
self.W = nn.Parameter(torch.randn(d_model, d_model))
Used in custom attention, research code..

In old gpt style models we also used nn.Parameter for positional encoding
self.pos_embed = nn.Parameter(torch.randn(max_len, d_model))


'''