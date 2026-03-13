'''
LayerNorm: Mean over feature dimension 
This makes normalisation independent of batch size which is what we want.

So normalisation happens per sample
input: (batch_size, seq_len, hidden_dim) and LayerNorm normalizes over this hidden_dim or the input

class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)

'''
import torch

def main():
    x = torch.tensor([
        [[1., 2., 3.],
         [4., 5., 6.]],
        
        [[7., 8., 9.],
         [10., 11., 12.]]
    ])  # shape: (2, 2, 3)  -> (Batch, W/row, H/col)-> 2 matrices (batch size = 2, but it (LayerNorm) doesnt rely on this), both have 2 rows 3 columns
    # layer normalization is applied to each input sequence individually. There are two major reasons for doing this. First, batch normalization is tricky to apply to sequence models (like transformers) where each input sequence can be a different length,
    # layer norm will apply over both matrices for every row (token vector) (same happens in transformers)
    layernorm = torch.nn.LayerNorm(normalized_shape=3) # features = 3
    y = layernorm(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape) # shape is same as input, no batch interaction
    print("Output:\n", y)

main()

'''
Running through the output:
Input shape: torch.Size([2, 2, 3])
Output shape: torch.Size([2, 2, 3])


Output:
 tensor([[[-1.2247,  0.0000,  1.2247],
         [-1.2247,  0.0000,  1.2247]],

        [[-1.2247,  0.0000,  1.2247],
         [-1.2247,  0.0000,  1.2247]]], 
         
         grad_fn=<NativeLayerNormBackward0>)

As you can see both matrices have been normalised

The Vector: For an input matrix where each row is a token (e.g., "The," "cat," "sat"), LayerNorm looks at the row of feature activations for just that one token.
The Calculation: It calculates the mean and variance using only the values within that single token's feature vector.
Independence: The normalization of the word "cat" does not use any information from the word "sat" or "the"

How it works?
For a single token vector -> [4, 5, 6], mean = 5, var = 2/3 
Normalise: [-1.22, 0.0, 1.22] (then scale + shift using learnable parameters (gamma, beta) same as BatchNorm)

Step 1: Pick one sentence in the batch (Sequence Independence).
Step 2: Pick one token (word) in that sentence (Token Independence).
Step 3: Calculate the mean and variance across all features (e.g., all 512 embedding values) for that specific token.
Step 4: Normalize that token's vector using its own statistics.

LayerNorm behaves IDENTICALLY in model.train() and model.eval()

Where does this happen?

Post-Normalization (Post-LN): In the original "Attention Is All You Need" paper, 
LayerNorm was placed after the residual connection of each sub-layer.
This configuration worked but required careful hyperparameter tuning and a specific "warm-up" phase for the learning rate to ensure stability.
x → Sublayer → Add → LayerNorm


Pre-Normalization (Pre-LN): Modern, deeper transformers typically use the Pre-LN setup, where normalization is applied before the attention and feed-forward sub-layers, inside the residual path. 
This approach results in much stabler gradients, faster and more reliable training (often without the need for learning rate warm-up), and allows for the creation of extremely deep models. 
x → LayerNorm → Sublayer → Add

Because every token is normalized using its own unique statistics, the model is not confused by variable sequence lengths or padding.

TLDR: it just speeds up + stabalises training using mean and variance, nothing fancy here.
'''

# How is LayerNorm implemented internally:
import numpy
error, gamma, beta = 0.01, 1, 0
class LayerNorm:
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_new = (x - mean) / numpy.sqrt(var + error)
        return gamma * x_new + beta