''' 
- measures how similar two vectors are (with scores from -1 (opposite) to 1 (identical))
- works by dividing the dot product of the vectors by the product of their magnitudes, essentially finding the angle's cosine. A smaller angle (closer to 0°) means higher similarity (score closer to 1)


Returns cosine similarity between  x1 and x2 computed along dim
dim = diension where cosine similaritty is computed (default:1)
epsilon = 1e-8 (small evalue to add to avoid division by 0)

Formula = x1 * x2 / max (|x1| * |x2|, epsilon)


How similar are two vectors by direction, not magnitude? (Are these two embeddings semantically similar?)
'''

import torch

def main():
    x = torch.tensor([[1., 0., 0.]])
    y = torch.tensor([[0.9, 0.1, 0.]])

    cos = torch.nn.CosineSimilarity(dim=1)

    sim = cos(x, y)
    print(sim) # tensor([0.9939])

main()

'''
Internal:

dot = (x * y).sum(dim)
norm_x = sqrt((x * x).sum(dim))
norm_y = sqrt((y * y).sum(dim))

cos_sim = dot / (norm_x * norm_y + eps) 

This is used in LLMs to calculate embedding similarity
query_embedding / document_embedding
↓
CosineSimilarity

Also used in re-ranking retrieved documnets in RAG:
LLM embedding → cosine sim → top-k → LLM reasoning

Why attention doesn't use cosine-similarity -> Attention needs Q*K.T
So direction doesn't matter -> Magnitude matters (confidence of how important one word is to another word)

def semantic_similarity(x, y):
    return cos(angle_between(x, y))

'''
