'''
Docstring for math_you_need_for_dl_code.multiheadattention

class torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)

We know attention = softmax((Q @ K.T) /  sqrt(d_model)) @ V

Let's take an example to see how this works.

Input -> "I am good"
3 words -> 3 vectors for each token, let's take size of each token = 2

[
[0.2 0.1] -> for "I"
[-0.9 0.4] -> for "am"
[0.7 0.8] -> for "good"
] dimension = [3,2] # 3 rows, 2 columns (seq_len, d_model)

so d_model = 2 (each word is being represented by 2 vectors)
and seq_len = 3 (there are 3 words in our vocab)

- Compute Key, Query, Values
PyTorch initialises random 
weights (3 * d_model, d_model)  -> 3 channels and d_model*d_model = [6,2]
and biases (3 * d_model, 1) -> 3 channels  and 3 * 2 = [6,1]

Visually
W = first two dots are for W_k, second two for W_q, last two for W_v (each has dimn 2,2 or d_model, d_model)
[
    . . 
    . .

    . .
    . .

    . .
    . .
]

B = first two for Bias_k, next two for Bias_q, last two for Bias_v (each has dimn 2,1)
[
    .
    .

    .
    .

    .
    .
]

Now we can compute K, Q, V (we take transpose so shapes match)
K = Input (3,2) * W_k (2,2).T + B_k (2,1).T = [3,2] + [1,2] = [3,2] is the shape for K
Q = Input * W_q.T + B_q.T = [3,2] shape of Q
V = Input * W_v.T + B_v.T = [3,2] shape of V

Now calculating attention
Q * K.T = [3,2] * [2,3] = [3,3], dividing by d_model and putting softmax -> dimensions don't change.

Our Q*K.T/d_model

[
. . .
. . .
. . .
]

After softmax, all scores are between 0 -> 1, this is called scores

      I   am  good
I     0.1 0.3 0.6
am    0.7 0.1 0.2
good  0.5 0.4 0.1


This is our softmax(Q@K.T/d_model) matrix which has attention scores which tells how much attention we need to pay for each word FROM each word.

now out =  scores[3,3] * V [3,2] = [3,3] matrix of output. This isn't final output.

PyTorch creates out_w = [d_model, d_model] and out_b = [d_model, 1]

final_out = out (3, 2) * out_w (2, 2).T + out_b (2, 1).T
          = (3, 2) + (1, 2)
          = (3, 2) is the final output of this layer -> which goes for further processing (normalise etc..)

also known as head1. MHA just does concat(head1, head2,...head_h) 

Every layer does this. Depends on number of heads.

torch.nn.MultiheadAttention does:
(“Which tokens matter in different ways at the same time?”)
- Projects inputs → Q, K, V
- Runs attention per head
- Concatenates heads + output projection

'''

import torch
def main():
    x = torch.randn(3, 2, 2) # (seq_len, batch, d_model)
    mha = torch.nn.MultiheadAttention(embed_dim=2, num_heads=2)
    out, attn_weights = mha(x, x, x)

    print(f"Output shape: {out.shape}") # final transformed embeddings
    print(f"Attention Weights: {attn_weights.shape}") # attention matrix (for debugging, LLMs do not use attention weights during inference)

main()

'''
Running through the output:
Output shape: torch.Size([3, 2, 2]) # output shape is 3,2 which is what we predicted!!
Attention Weights: torch.Size([2, 3, 3])


Revising process:

Q = X @ W_q
K = X @ W_k
V = X @ W_v

Q → (seq_len, batch, H, D)
K → (seq_len, batch, H, D)
V → (seq_len, batch, H, D)

softmax(Q_h K_hᵀ / sqrt(D)) @ V_h

Concat(head_1, ..., head_H)
→ shape: (seq_len, batch, E)

out = concat @ W_o

Causal Mask in MHA:
attn_mask = torch.triu(torch.ones(L, L), diagonal=1)
attn_mask = attn_mask.bool()
key_padding_mask = (tokens == PAD_ID)


attn_mask → blocks (i → j)
key_padding_mask → blocks tokens entirely
'''

# How it works inside the hood
'''
class MultiHeadAttention:
    def forward(self, x):
        Q, K, V = linear(x)
        split heads
        for each head:
            head = attention(Q, K, V)
        concat heads
        return output_projection
'''