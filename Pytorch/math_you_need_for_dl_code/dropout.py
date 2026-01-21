'''
During training, randomly zeroes some of the elements of the input tensor with probability p
We do this to reduce overfitting during training since we don't need all neurons for training everytime. Also makes training faster.

To compensate for the fact that we are dropping out some neurons we also need to scale the outputs by factor of 1/1-p during training.
This means during evaluation the module simply computes an identity function (y = x) and we don't have to scale down or up when it comes to testing, we do it in the training itself.

Parameters:
p (float) – probability of an element to be zeroed. Default: 0.5
inplace (bool) – If set to True, will do this operation in-place. Default: False

Writing this mathematically as a function
y = 0 with probability p and xi / (1-p) with probability (1-p) # scaling (So the expected value stays the same)

By default p = 0.5 hence scale factor will be 1 / (1-0.5) = 2:
- So we zero out randomly 50% of the neurons / inputs during training and with the remaining neurons / inputs we multiply them by 2. Makes sense.

'''

import torch
def main():
    input_data = [[1., 2., 3.],
                 [4., 5., 6.]]   # shape: (rows/batch=2, columns/features=3) 
    input_data_tensor = torch.Tensor(input_data)
    dropout = torch.nn.Dropout(p=0.5)

    after_dropout = dropout(input_data_tensor)

    print(f"Shapes of input and output remain the same: {input_data_tensor.shape}, {after_dropout.shape}")
    print(f"Input : {input_data}")
    print(f"Output : {after_dropout}")


main()

'''
Running through the output:
Shapes of input and output remain the same: torch.Size([2, 3]), torch.Size([2, 3])


Input : [
            [1.0, 2.0, 3.0], 
            [4.0, 5.0, 6.0]
        ]
Output : tensor(
        [
            [ 0.,  4.,  0.],
            [ 0.,  0., 12.]
        ])
As we can see, 50% of the inputs have been zeroed out randomly (running this multiple times will lead to different results (different inputs being zeroed out as shown below)) and non-zeroed out inputs are scaled by 2.


After running it again (different results):

Output : tensor(
        [[ 0.,  0.,  0.],
        [ 8., 10., 12.]])

Training mode (model.train())
- Random masking
- Scaling applied

Eval mode (model.eval())
- Dropout is DISABLED
- Output = input (identity)

In Transformers, dropout is used in:
- Attention weights
- FFN hidden layers
- Residual connections

But NOT usually on:
- Token embeddings (or very lightly)
- Final logits

Modern large LLMs often reduce or remove dropout
because massive data already regularizes well

Where is it placed while training a neural network?
model = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
     # HERE
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(256, 10)
)

'''

# How dropout class is implemented:
import random
class Dropout:
    def __init__(self, p):
        self.p = p

    def forward(self, x, training=True):
        if not training:
            return x
        mask = (random.rand(x.shape) > self.p)
        return x * mask / (1 - self.p)



