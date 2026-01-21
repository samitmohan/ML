'''
We need positional encoding in Transformer models because their parallel processing mechanism (self-attention) doesn't inherently understand word order, which is crucial for meaning in language
Positional encodings add unique position information to each word's embedding, allowing the model to differentiate words like "cat" and "dog" in "The cat chased the dog" versus "The dog chased the cat" and understand relative positions.

Example:

Let's take the words good boy
good position = 0, boy position = 1
and d_model = 4

Positional Encoding has dimension -> [d_model, 1]

Formulas:
PE(p, 2i) = sin(p / 10000 ^ 2i/d)
PE(p, 2i+1) = cos(p / 10000 ^ 2i/d)

p is the position, d = dimension, i = list (of length d_model)
i
[0
 1
 2
 3]

i / 2 = [
        0
        0
        1
        1
        ]

Final position encoding for word 'good' 
i = 0, p = 0 -> plug in values in formula -> sin(0/10000^2*0/4) and 
i = 1, p = 0 -> cos(0/10000^2*0/4)

same for all positions and same for the word 'boy'

good(d_model,1) + boy(d_model,1) = final position vector = (d_model, 1)

After this you can add this position embedding to your original input vector tokens and pass it to attention:
embeddings = token_embeddings + sinusoidal_pos_enc(seq_len, d_model)[:seq_len]
'''
import torch, math

def sinusoidal_pos_enc(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle = pos / (10000 ** (2*(i//2)/d_model))
    enc = torch.zeros(seq_len, d_model)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])
    enc[:, 1::2] = torch.cos(angle[:, 1::2])
    return enc  # (seq_len, d_model)

# Example: Encode positions for a sequence
seq_len = 10      # 10 tokens in sequence
d_model = 512     # Embedding dimension

pos_enc = sinusoidal_pos_enc(seq_len, d_model)
print(pos_enc.shape)  # torch.Size([10, 512])

