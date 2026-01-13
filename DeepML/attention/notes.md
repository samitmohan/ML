## What is Attention 

Sentence -> Each word/token has query, key, value vectors.

The query from a word is compared with keys of all other words to determine how much attention is needed for each word.

The values are aggregated based on these 'attention weights'

## Generating query, key, value vectors

- Each input token is embedded into vec and multiplied by wieght matrices Wq, Wk, Wv to get the query, key, value vectors.
- Attention scores between token are calculated using 
    - (dot(query, key)) vectors
- These scores are scaled by square root of dimension of key vectors to prevent dot products from getting too large (leads to vanishing gradients)
- Softmax is applied -> prob that sum to 1.

These weights are used to take weighted avg of value vectors.
Result = output of attention mechanism for each posn. {self-attention since each posn attends to all posns in same seq}

```math
x = input embedding for token (let d be the dimension of this vector)
q = x dot Wq
k = x dot Wk
v = x dot Wv

A_score = (q dot transpose(k)) / square_root(d)

```

Transformers use multi head attention
 - Instead of computing single attention mechanism, the model splits input into multiple heads -> each computing their own attention. 
 - Outputs from each head are concated then linearly transformed again to get final output.


Why? To capture different types of relationships in the data, this also allows the model to attent to multiple relationship in the data at once, making it incredibly flexible and expressive 
Most common Type of attention is self attention where Q, K, V all come from same input sequence. 
Meaning? Every token  in sequence can attend to every other token including itself.

This bidirectional awareness is what Transformers are good for (capturing context) Unlike RNN/LSTM which can only look backward or forward.


q: what the model is looking for (query matrix)
k: what the model can offer (repr of all tokens) 
v: actual content of tokens to be weighted and combined.

Steps:
- Compute dot(q, transpose(k)) resulting in score matrix
- Scale by dividing by square_root(dimension)
- Softmax along last dimension (keys)
- Weighted sum: multiply by V obtaining final attention output

Attention in Transformer Architecture.
Transformer has Encoder, Decoder stacked with multiple layers.

Encoder
- Uses self attention to process input sequence (sentence in english)
- Each layer refines representation

Decoder
- Uses Masked Self Attention to process output sequence (translate in french) 
  - Masked: Only attends to prevs posn (Doesn't look at future token of next word during training, tries to predict it)

- Uses Cross Attention to connect decoder to encoder letting output attend to the input (Aligning french words with english words)

```text
Think of Attention as a spotlight operator in a theater. The Query is the director saying, “Focus on the lead actor!” The Keys are all the actors on stage, and the Values are their lines. The spotlight (Attention) adjusts its beam based on who’s most relevant, illuminating the scene dynamically as the play unfolds.
```

## Multi Head Attention

What do you mean splitting work across multiple heads.
Each head focusses on different aspects of input
    - Grammar
    - Meaning
    - Context

Together they give model a better understanding.

In the original dot-product attention, the raw scores can become very large if the dimensionality of the query and key vectors (d) is high which causes softmax function to produce extremely small gradients (no learning)

Here,(d), is the dimensionality of the key vectors. Dividing by ensures that the dot product are scaled appropriately, preventing the softmax function from saturing

## Why does it work

Single head attention is great, but it's limited. It computes one set of attention weights and mixes all the information into single output. That's like trying to hear every instrument in symphony with one ear it works, but you miss the layers. Multi-Head Attention says, "Why settle for one prespective?" By running attention multiple times in parallel. each with its own lens (or "head"), the model captures diverse relationships in the data-like how " it" refers to "cat" in one head, while another head notices the verb tense.

Plus, its still parallelizable (unlike RNN's), so its fast. More heads = more insights, without slowing things down.

## Intuition for Multi Head
Multi Head attention steps in like a team of detectives, each with their own flashlight and intelligence

   - Detective 1 : Focuses on the players (“thief” → “guard”). Who’s involved? They spot the key characters in this drama.
   - Detective 2: Tracks the action and timing (“escaped” → “dozed off”). When did it happen? They link the verbs to figure out the sequence.
   - Detective 3: Sniffs out the cause-and-effect (“after” ties it all together). Why did it work? They catch the sneaky logic of the heist.

The result? A richer, multi-layered picture of the heist—way more detailed than what one lone detective could crack on their own.
