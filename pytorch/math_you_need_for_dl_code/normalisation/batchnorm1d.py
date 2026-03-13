'''
Docstring for math_you_need_for_dl_code.normalisation.batchnorm1d

class torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
Applies Batch Normalization over a 2D or 3D input.

Why even apply batch normalization, more importantly what even is batch normalisation
More importantly : What is normalisation

It ensures that the outputs of each batch we send (as inputs) have a consistent scale and distribution as data flows through the network.

In simpler words -> stabilizes and accelerates the training of deep neural networks by noramlising features

For each feature dimension, independently: it normalises input x by subtracting mean(entire batch) and dividing it by std variance of batch which makes sense -> inputs are normalised and more stable wrt to the entire batch.

x_new = x - mean_batch / sqrt(variance) + some error
Then it applies the affline parameters:
y = gamma * x_new + beta (here gamma(scale) and beta)shift are learned parameters via training)

What it expects:
input: fully connected layers of shape (batch_size, num_features)
Normalization is done over batch dimension.
'''

import torch
def main():
    x = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]]) # here number of features = columns = 3 (think of features as age, name, job in a dataset)
    
    bn = torch.nn.BatchNorm1d(num_features=3)

    y = bn(x)
    print(f"Input: {x}")
    print(f"Output: {y}")
    # shapes are preserved (both have (2,3))
    print(bn.weight)  # γ (gamma)
    print(bn.bias)    # β (beta)


main()

'''
Running through the output:
Input: tensor(
        [[1., 2., 3.],
        [4., 5., 6.] ])


Output: tensor(
        [[-1.0000, -1.0000, -1.0000],
        [ 1.0000,  1.0000,  1.0000]]
        , grad_fn=<NativeBatchNormBackward0>)

As you can see the output is now normalised.

You can see the bn.weight (gamma) and bn.bias(beta) {the learnable parameters}
Gamma: tensor([1., 1., 1.], requires_grad=True)
Beta: tensor([0., 0., 0.], requires_grad=True)

Internally:
x =
[[1, 2, 3],
 [4, 5, 6]]

mean = [2.5, 3.5, 4.5]
var  = [2.25, 2.25, 2.25]

normalise = (x - mean) / sqrt(var + eps)
scale and shift = γ * normalized + β (initially γ = 1 and β = 0)
print(bn.weight)  # γ (gamma)
print(bn.bias)    # β (beta)

During training
- Uses current batch mean & variance
- Updates running_mean and running_var

During evaluation
- Uses running_mean and running_var

With BN → distributions stay centered & scaled
Where is it placed? Linear → BatchNorm → ReLU
Benefits: - Faster convergence and Smoother optimization landscape

BatchNorm is not used in Transformers since it breaks with variable sequence lengths
Instead in Transformers we use LayerNormalisation which instead of applying this thing to Batches of data, applies to every layer.

Why does batchnorm break in transformers?
Transformers operate under these conditions:
- Variable sequence lengths
- Autoregressive decoding (token by token)
- Small or batch-size = 1 at inference (one token is sent at a time)
step 1 → token 1
step 2 → token 2
step 3 → token 3
Each step:
- Batch size = 1
- Sequence length grows
- BatchNorm cannot compute meaningful statistics here.
Since batchnorm computes mean over batch, but in transformers batch = 1 so it results in unstable normalisastion & doesn't really make sense.



Order matters per token, not across batch
To fix this we use LayerNorm

'''

# How batchnorm is implemented internally
import numpy as np
error, gamma, beta = 0.01, 1, 0
class BatchNorm1d:
    def forward(self, x, training=True):
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            update_running_stats(mean, var)
        else:
            mean = running_mean
            var = running_var

        x_new = (x - mean) / np.sqrt(var + error)
        return gamma * x_new + beta