'''
Docstring for math_you_need_for_dl_code.transformer_decoder

Let's go through the entire Decoder process

High level overview::

nn.TransformerDecoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation=relu,
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False
)

It is a prewired composition of:
- masked multihead self-attention
- cross (encoder–decoder) attention
- feed forward network
- residual connections
- layernorm
- dropout

Input shapes:
- tgt (decoder input): (tgt_seq_length, batch, d_model)
- memory (encoder output): (src_seq_length, batch, d_model)

Output shape:
- same as tgt: (tgt_seq_length, batch, d_model)

---------------------------------

Decoder architecture (pre-norm variant, norm_first=True):

x (decoder input embeddings with positional encoding added)
↓
Masked Self-Attention (causal)
↓
Dropout
↓
Add (residual)
↓
Cross-Attention (attend to encoder memory)
↓
Dropout
↓
Add (residual)
↓
FeedForward Network
↓
Dropout
↓
Add (residual)

As functions (pre-norm):

x = x + Dropout(SelfAttention(LayerNorm(x)))        # causal self-attn
x = x + Dropout(CrossAttention(LayerNorm(x), memory))
x = x + Dropout(FFN(LayerNorm(x)))

---------------------------------

Connecting all blocks with an example:

Sentence: "I am samit"

Tokenizer → token IDs:
"I am samit" → [t0, t1, t2]
seq_length = 3
batch_size = 1

Model parameters:
d_model = 8
nhead = 2 → head_dim = 4
dim_feedforward = 32
batch_first = False → tensors are (seq_len, batch, d_model)
norm_first = True (pre-norm, used in modern Transformers / LLMs)

---------------------------------

Step 0: Tokenization
Text → tokenizer → token IDs
Decoder input receives shifted tokens during training (teacher forcing).

---------------------------------

Step 1: Embedding + positional encoding

token_emb = token_embedding(token_ids)
pos_emb   = positional_embedding(positions)
X = token_emb + pos_emb

Shape:
X → (3, 1, 8)

X[0] = embedding for "I"
X[1] = embedding for "am"
X[2] = embedding for "samit"

---------------------------------

Step 2: Masked Self-Attention (Causal)

x_norm = LayerNorm(X)

Q, K, V = Linear(x_norm)

After splitting heads:
Q, K, V → (batch, nhead, seq_len, head_dim) = (1, 2, 3, 4)

Apply causal mask:
- token i can only attend to tokens ≤ i
- prevents information leakage from future tokens

Compute:
scores = (Q · Kᵀ) / sqrt(head_dim)
scores[j > i] = -inf
weights = softmax(scores)

out_self = weights · V

Concatenate heads + output projection:
out_self → (3, 1, 8)

Residual:
X = X + out_self

---------------------------------

Step 3: Cross-Attention (Encoder–Decoder Attention)

Purpose:
- Decoder tokens attend to encoder outputs ("memory")
- This is how the decoder conditions on source text or retrieved context

x_norm = LayerNorm(X)

Q = Linear(x_norm)           # from decoder
K = Linear(memory)           # from encoder
V = Linear(memory)

Shapes:
Q → (1, 2, 3, 4)
K,V → (1, 2, src_seq_len, 4)

Compute:
scores = (Q · Kᵀ) / sqrt(head_dim)
weights = softmax(scores)
out_cross = weights · V

Concatenate heads + projection:
out_cross → (3, 1, 8)

Residual:
X = X + out_cross

---------------------------------

Step 4: Feed Forward Network (FFN)

x_norm = LayerNorm(X)

hidden = Linear1(x_norm)          # (3, 1, 32)
hidden = activation(hidden)       # GELU / SwiGLU
out_ffn = Linear2(hidden)         # (3, 1, 8)

Residual:
X = X + out_ffn

---------------------------------

Final output:

Shape:
X → (3, 1, 8)

Interpretation:
- Each decoder token embedding is now informed by:
  - previous decoder tokens (via causal self-attention)
  - encoder tokens (via cross-attention)
  - non-linear feature transformation (via FFN)

Gradients flow through all paths:
FFN ← attention ← embeddings ← tokenizer

---------------------------------

Modern LLM usage notes:

- Decoder-only LLMs (GPT, LLaMA):
  - remove cross-attention
  - keep masked self-attention + FFN
- Encoder–Decoder models (T5, BART):
  - use full decoder with cross-attention
- Production models:
  - use pre-norm + RMSNorm
  - use FlashAttention / SDPA
  - use SwiGLU FFNs
  - use KV caching during inference

This class is educational and correct,
but large-scale LLMs reimplement it manually for performance.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sentence = ["I", "am", "samit"]
token_to_id = {"I": 0, "am": 1, "samit": 2}

tgt_ids = torch.tensor([token_to_id[w] for w in sentence])
src_ids = torch.tensor([0, 1, 2])  # pretend encoder input

tgt_len = len(tgt_ids)
src_len = len(src_ids)
batch_size = 1
d_model = 8
nhead = 2
head_dim = d_model // nhead
dim_ff = 32

# Embeddings
embedding = nn.Embedding(10, d_model)

x = embedding(tgt_ids).unsqueeze(1)       # decoder input (tgt)
memory = embedding(src_ids).unsqueeze(1)  # encoder output (memory)

print("\nDECODER INPUT x")
print(x.shape)
print(x)

print("\nENCODER MEMORY")
print(memory.shape)
print(memory)

# ---- helpers ----
def split_heads(t):
    return t.view(-1, batch_size, nhead, head_dim).permute(1, 2, 0, 3)

def causal_mask(L):
    return torch.tril(torch.ones(L, L)).bool()

# ---- Sublayer 1: Masked Self-Attention ----
ln1 = nn.LayerNorm(d_model)

Wq = nn.Linear(d_model, d_model, bias=False)
Wk = nn.Linear(d_model, d_model, bias=False)
Wv = nn.Linear(d_model, d_model, bias=False)
Wo = nn.Linear(d_model, d_model, bias=False)

x_norm = ln1(x)
Q = split_heads(Wq(x_norm))
K = split_heads(Wk(x_norm))
V = split_heads(Wv(x_norm))

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
scores = scores.masked_fill(~causal_mask(tgt_len), float("-inf"))

weights = F.softmax(scores, dim=-1)
out_self = torch.matmul(weights, V)

out_self = out_self.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size, d_model)
x = x + Wo(out_self)

print("\nAFTER MASKED SELF-ATTENTION")
print(x)

# ---- Sublayer 2: Cross-Attention ----
ln2 = nn.LayerNorm(d_model)

Wq_c = nn.Linear(d_model, d_model, bias=False)
Wk_c = nn.Linear(d_model, d_model, bias=False)
Wv_c = nn.Linear(d_model, d_model, bias=False)
Wo_c = nn.Linear(d_model, d_model, bias=False)

x_norm = ln2(x)
Q = split_heads(Wq_c(x_norm))
K = split_heads(Wk_c(memory))
V = split_heads(Wv_c(memory))

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
weights = F.softmax(scores, dim=-1)

out_cross = torch.matmul(weights, V)
out_cross = out_cross.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size, d_model)

x = x + Wo_c(out_cross)

print("\nAFTER CROSS-ATTENTION")
print(x)

# ---- Sublayer 3: FFN ----
ln3 = nn.LayerNorm(d_model)

fc1 = nn.Linear(d_model, dim_ff)
fc2 = nn.Linear(dim_ff, d_model)

x_norm = ln3(x)
ff = fc2(F.gelu(fc1(x_norm)))

x = x + ff

print("\nFINAL DECODER OUTPUT")
print(x.shape)
print(x)

'''
Decoder has two attentions: masked self-attn (causal) + cross-attn (attend to encoder).
Decoder enforces time flow in self-attn; encoder uses full (non-causal) self-attention.
Decoder is used in: Encoder–Decoder models (e.g., T5, original Transformer for translation).
Decoder-only LLMs (they omit cross-attention and only use masked self-attention).
Cross-attention is the core conditioning mechanism in seq2seq and RAG-style pipelines (retrieval results as memory).
'''