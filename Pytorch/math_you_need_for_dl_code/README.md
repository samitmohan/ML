# torch.nn

```text
Embedding
→ [ LayerNorm
    → Attention (Linear + Softmax)
    → Residual
    → LayerNorm
    → FFN (Linear → GELU → Linear)
    → Residual ] × N
→ Linear
→ CrossEntropyLoss
```