'''
Docstring for math_you_need_for_dl_code.crossentropyloss

This is the loss function used to train almost all LLMs.

Measures how wrong were the predicted probabilities for the correct class?

It's a combination of LogSoftmax + Negative Log Likelihood
Math:
-log(softmax)

Given:
- model outputs scores (logits)
- correct class index

It answers:
“How much probability did you assign to the correct answer?”

- High probability → low loss
- Low probability → high loss
'''
import torch

import torch

def main():
    logits = torch.tensor([[2.0, 1.0, 0.1]])  # (batch, num_classes)
    target = torch.tensor([0])               # correct class index (batch_size,)

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(logits, target)
    print(loss) # tensor(0.4170) (Our goal is to minimise this using backprop)

main()


'''
Running through the output
Step1: log(softmax)
log_probs = log(softmax(logits))

Step2: picking correct class
correct_log_prob = log_probs[range(batch), target]

Step3: negate + average
loss = -mean(correct_log_prob)

CrossEntropyLoss expects raw logits it applies softmax automatically

In LLMs:
Given:
(batch, seq_len, vocab_size)  ← logits
(batch, seq_len)              ← target token IDs

We reshape:
logits = logits.view(-1, vocab_size) # as 1 list
targets = targets.view(-1) # as 1 list
loss = CrossEntropyLoss(logits, targets) # apply loss fn

For each token:

“How surprised was the model by the correct next token?”

- Predicts common token → low loss
- Predicts rare token → higher loss
- Predicts wrong token → very high loss

Average over all tokens.

How is it implemented:

def loss(logits, correct_class):
    probs = softmax(logits)
    return -log(probs[correct_class])


'''