'''
Docstring for math_you_need_for_dl_code.embedding

learnable lookup table
Map an integer ID → a vector.

nn.Embedding(num_embeddings, embedding_dim)

it stores a matrix: (weight): (num_embeddings, embedding_dim) (Each row is the vector representation of one discrete symbol)

Neural Networks can't operate on integers directly. 
"hello" → 10321
"world" → 45

These words have no meaning geometrically.

Embedding turns token_id → continuous vector

'''

import torch

def main():
    embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=4)

    token_ids = torch.tensor([1, 3, 7]) # [1, 3, 7]      → shape (3,) input
    # lookup → add embedding_dim


    vectors = embedding(token_ids)

    # (3 tokens, 4 features)

    print(vectors.shape) # torch.Size([3, 4]) {Embedding adds one dimension at the end.}
    print(vectors) 
    '''
    tensor([[ 0.7213, -0.7668,  0.2204, -1.0903],
        [-1.6225,  0.6308, -1.6110, -0.7528],
        [-1.1358, -2.2639, -0.7726, -1.6766]], grad_fn=<EmbeddingBackward0>)
    '''

main()


'''
Running through the output

Each row of the embedding matrix is a trainable parameter.

During backprop:
- Only rows used in the batch get gradients
- Other rows are untouched

How is it used in LLMs
Token embedding:
token_embed = nn.Embedding(vocab_size, d_model)

Input:(batch, seq_len)
Output:(batch, seq_len, d_model)

This is the very first layer of an LLM.

Even for positional encoding (GPT2, BERT):

pos_embed = nn.Embedding(max_len, d_model)


How it works internally:

class Embedding:
    def __init__(self):
        self.table = random_matrix()

    def forward(self, ids):
        return self.table[ids]

How info flows:
Text
↓
Tokenizer
↓
Token IDs (integers)
↓
nn.Embedding
↓
Vectors

Let's say your sentence is ["hello", "my", "name", "is"]
Which is mapped to numbers:
{
  "hello": 1532,
  "my": 212,
  "name": 784,
  "is": 318
}

So token_ids = [1532, 212, 784, 318]

Now embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)

This creates a table embedding.weight.shape = (10, 4)
ID → vector
0  → [ ... ]
1  → [ ... ]
2  → [ ... ]
...
9  → [ ... ]

hello → vector
my    → vector
name  → vector
is    → vector


Just makes it into a vector

vocab = {
  token_id: learned_vector
}

sentence = [id1, id2, id3]

vectors = [vocab[id] for id in sentence]

'''