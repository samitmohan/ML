'''
output = attention(Q, K, V)


PyTorch’s fused / optimized implementation of the classic attention core: 
compute softmax((Q @ K.T) / d_model) @ V with optional masking, dropout, and a few performance features (GQA, fused kernels)


How it works:
Given tensors Q, K, V:


'''
import torch
import torch.nn.functional as F

def main():
    Q = torch.randn(2, 3, 4)  # (batch, tokens(rows), dimensions(columns))
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 4)

    out = F.scaled_dot_product_attention(Q, K, V)

    print(f"Output shape: {out.shape}")
    print(f"Output : {out}")
          
main()

'''
Running through the output
Output shape: torch.Size([2, 3, 4]) # same as (Q,K,V) (2 batches, 3 rows, 4 columns)
Output : tensor([
        [
            [-1.1798,  0.0962, -0.4744,  0.3941],
            [ 0.1781,  0.2997, -0.0952, -0.4089],
            [-1.3043,  0.0902, -0.5171,  0.4701]],

            [[-0.2180,  0.4441,  1.9291, -1.0765],
            [ 0.2576,  0.2707,  0.9700,  0.7783],
            [ 0.0606,  0.2904,  1.4156, -0.0605]]])
'''

'''
Can also use masking (token t can only attend to tokens ≤ t)
- Block future tokens (causal) 
- Masked positions get -inf.

Can also set dropout (at training):

out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=(0.1 if model.training else 0.0),
    is_causal=True
)

LLMs do:

Linear → Q, K, V
↓
scaled_dot_product_attention
↓
Linear

'''
# How is this implemented:
import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention( Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, training=True):
    # shape: Q, K, V: (batch, heads, seq_len, head_dim)

    B, H, seq_len, d_model = Q.shape
    _, _, S, _ = K.shape

    # dot product -> similarity score
    # (B, H, L, D) @ (B, H, D, S) → (B, H, L, S)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # scale
    scores = scores / math.sqrt(d_model)

    # causal mask apply using torch.tril -> For each query position i; token i can attend to tokens j where j ≤ i
    '''
    Visually
    i\j  0  1  2  3  4
    0   1  0  0  0  0
    1   1  1  0  0  0
    2   1  1  1  0  0
    3   1  1  1  1  0
    4   1  1  1  1  1
    ❌ Look ahead
    ✅ Look backward (and self)
    '''
    if is_causal:
        causal_mask = torch.tril( torch.ones(seq_len, S, device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf")) # fill with -inf

    # attention mask: Which tokens are valid to attend to for this input?
    '''
    Input:  [Hello, world, <PAD>, <PAD>]
    Mask:   [1,     1,      0,     0]
    '''
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask # normal default (no padding)

    # softmax
    attn_weights = F.softmax(scores, dim=-1)

    # dropout
    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Weighted sum of values
    # (B, H, L, S) @ (B, H, S, D) → (B, H, L, D)
    output = torch.matmul(attn_weights, V)

    return output


'''
This is the heart of attention which is used in Transformers, this is how similarity / how much each word should 'pay attention' to other words is computed.
Word embeddings -> similar words are together (king and queen are together in vector space) -> hence this calculates simiarity between king and queen by using dot product of 3 vectors (Q,K,V)
If dot product -> large -> they are in the same direction -> similar, else they aren't.

Q,K,V are parameters -> Query, Key, Value

Query (Q): What I am looking for?
This is like the search term you type into a search engine or the question you ask. It represents the current word or token's need for information from other words in the sequence.
Key (K): What information do I offer?
This is like the title, keywords, or tags associated with a webpage or item in a database. Each key is used to determine how relevant its corresponding information (value) is to the query.
Value (V): What information do I contain?
This is the actual content or full story associated with a key, which is retrieved if the query and key match well. In the self-attention mechanism, the values of all words are combined, weighted by their relevance scores to the query. 

Consider the sentence: "The cat sat on the mat because it was tired". 
When the model processes the word "it" and wants to understand what "it" refers to:
The Query (Q) for "it" asks, "What is a singular, non-human noun that can be tired?".
This query is compared to the Keys (K) of all other words.
The Key for "cat" might have information suggesting it's an animal and can be tired, resulting in a high similarity score. The Key for "mat" would have a low score.
The model uses this high score to pay more attention to the Value (V) of "cat," which holds the rich numerical meaning of the word. The value for "mat" is mostly ignored.
The final representation of "it" is then updated to include the relevant context from "cat," allowing the model to correctly understand the pronoun

'''

# Another simpler implementation
# attention.py
import torch, math
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: (batch, heads, seq_len, d_k)
    scores = Q @ K.transpose(-2,-1) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights

# tiny demo
B, H, S, d = 2, 1, 4, 8
Q = torch.randn(B,H,S,d); K = torch.randn(B,H,S,d); V = torch.randn(B,H,S,d)
out, w = scaled_dot_product_attention(Q,K,V)
print(out.shape, w.shape)
