'''
Docstring for math_you_need_for_dl_code.transformer_encoder

Let's go through the entire Encoder process

High level overview::

nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation=relu,
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False
)

It is a prewired composition of multihead attention, feed forward network, residual connection, layernorm and dropout

Input shape: (seq_length, batch, dimension_model) or (batch, seq_length, dimension_model) if batch_first=True 
Output shape: same

It works exactly like Transformer Encoder model with architecture:

x (input embeddings with positional encoding added (assumption))
↓
Self-Attention
↓
Dropout
↓
Add (residual)
↓
LayerNorm
↓
FeedForward
↓
Dropout
↓
Add (residual)
↓
LayerNorm

As functions:
x = LN(x + Dropout(SelfAttention(x)))
x = LN(x + Dropout(FFN(x)))



--- 

Connecting all blocks with an example:
"I am samit" -> Tokeniser -> [t0, t1, t2] IDs corresponding to ["I", "am", "samit"] so seq_length = 3, batchsize=1
Hidden model size = d_model = 8, number of heads = 2, thus head_dimension = d_model/nhead = 4
FFN size = dim_feedforward = 32 (in GPT it's 2048)
Use batch_first=False default, so tensors are of shape (seq_len, batch, d_model) = (3,1,8)
Use norm_first=True (pre-norm (used in Transformer++ models, stablises gradients more than post norm))
No padding / Causal mask (This is encoder)

Step 0: Tokenization -> token IDs (text -> tokenizer -> ids)
"I am samit"  ->  token_ids = [1532, 212, 784]  (these are feeded into nn.Embedding) to create vector embeddings of these tokens.

Step 1: Embedding + positional encoding = input tensor X
token_emb = token_embedding(token_ids)
pos_emb = pos_embeddings(positions)
X = token_emb + pos_emb # shape: (3,1,8)

X[0] = vector for "I"    shape (1, 8)
X[1] = vector for "am"   shape (1, 8)
X[2] = vector for "samit"shape (1, 8)

Step 2: TransformerEncoderLayer (with norm_first=True)
- x = x + self_atten(LayerNorm(x))
- x = x + FFN(LayerNorm(x))

Breaking this down:
First we compute LayerNorm -> x_norm = Layernorm(X)
LayerNorm normalizes each token vector across its 8 features: mean and var computed per token (per position), then x_norm = (x - μ)/sqrt(var+eps) scaled by learnable γ and shifted by β. No shape change.

Then nn.MultiheadAttention is used that does three linear projects from d_model -> d_model:
- For each token we compute Q_i = W_q * X_norm + b_q. Same for K_i and V_i

After splitting into heads reshape to (seq_length, batch, nhead, head_dim) -> transpose to common attention shape (batch, nhead, seq_length, head_dim)
Q.shape = (B, nhead, sequence_length, head_dim) = (1, 2, 3, 4)
K.shape = (1, 2, 3, 4)
V.shape = (1, 2, 3, 4)

Then we do scaled dot product attention per head. score_i_j = softmax((Q_i*Kj)/sqrt(head_dim)) where i = query token position and j = key position
Then weighted sum with V -> out = V_i * score
Do this for every head, every token position.
Concatenate heads and output projection -> final shape same as input -> (3,1,8) -> Apply lienar projection W_o (d_model, d_model) to mix head outputs

Apply dropout optionally (p = 0.5) ~ drops 50% of neurons for stable training and to reduce overfitting

Then add residual (add original X)
X = X + dropout(mha(ln(x))) # share preserved (3,1,8)

Then we have the FFN.
- LayerNorm
X_norm2 = LayerNorm(X) # shape still (3,1,8)
hidden = Linear1(X_norm2) # shape = (3,1,dim_feedforward) = (3,1,32) projecting to higher space
hidden_activation = activation(hidden) # RELU/SWIGLU
hidden_drop = Dropout(hidden_activation)
out_ffn = Lienar2(hidden_dropout) # shape : (3,1,8) again (downsample, project back to original dimension)

Again add residual to this:
X = X + Dropout(out_ffn)

Final output of the layer
Shape remains (seq_len, batch, d_model) = (3,1,8):
X[0] is the updated embedding for token "I", now informed by "am" and "samit".
X[1] updated embedding for "am".
X[2] updated embedding for "samit".
All values are differentiable; gradients will flow back through FFN, attention, LayerNorm, embeddings to update parameters.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sentence = ["I", "am", "samit"]
token_to_id = {"I": 0, "am": 1, "samit": 2}

token_ids = torch.tensor([token_to_id[w] for w in sentence])
seq_len = len(token_ids)
batch_size = 1
d_model = 8
nhead = 2
head_dim = d_model // nhead
dim_ff = 32

# Embedding
embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model)
x = embedding(token_ids)              # (seq, d_model)
x = x.unsqueeze(1)                    # (seq, batch, d_model)

print("\n TOKEN EMBEDDINGS")
print("x shape:", x.shape)
print(x)

# LayerNorm (Pre-Norm)
ln1 = nn.LayerNorm(d_model)
x_norm = ln1(x)

print("\nAFTER LAYERNORM (PRE-ATTN)")
print(x_norm)

# QKV Projection
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x_norm)
K = W_k(x_norm)
V = W_v(x_norm)

print("Q shape:", Q.shape)
print(Q)

# reshape for heads
def split_heads(t):
    return t.view(seq_len, batch_size, nhead, head_dim).permute(1, 2, 0, 3)

Qh = split_heads(Q)
Kh = split_heads(K)
Vh = split_heads(V)

scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(head_dim)

print("scores shape:", scores.shape)
print(scores)

weights = F.softmax(scores, dim=-1)

print("\nATTENTION WEIGHTS (SOFTMAX)")
print(weights)

attn_out = torch.matmul(weights, Vh)

print("\nATTENTION OUTPUT PER HEAD")
print(attn_out)

# Concatenate Heads
attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
attn_out = attn_out.view(seq_len, batch_size, d_model)

W_o = nn.Linear(d_model, d_model, bias=False)
attn_out = W_o(attn_out)

# Residual
x = x + attn_out

print("\nAFTER ATTENTION + RESIDUAL")
print(x)

# FFN (Pre-Norm)
ln2 = nn.LayerNorm(d_model)
x_norm = ln2(x)

fc1 = nn.Linear(d_model, dim_ff)
fc2 = nn.Linear(dim_ff, d_model)

ff_hidden = fc1(x_norm)
ff_hidden_act = F.gelu(ff_hidden)
ff_out = fc2(ff_hidden_act)

print("\nFFN HIDDEN (AFTER GELU)")
print(ff_hidden_act)

# Residual
x = x + ff_out

print("\nFINAL OUTPUT OF ENCODER LAYER")
print("output shape:", x.shape)
print(x)


'''
The Transformer++ Model
Modern LLM engineering practices

Pre-norm (LayerNorm before sublayer) is preferred for stability in deep stacks.

Large LLM implementations typically reimplement the encoder/decoder block manually for performance: fused QKV projections, scaled_dot_product_attention using FlashAttention or other memory-efficient kernels, and custom FFNs (SwiGLU / gated activations).

Many state-of-the-art LLMs replace LayerNorm with RMSNorm or small variants for efficiency/stability.

Dropout is often reduced or removed for very large models (regularization comes from scale and data).

FFNs in modern LLMs often use gated activations (SwiGLU) with different hidden-size scaling (e.g., 2× or 4× d_model with different parameter efficiency).
'''