# https://www.deep-ml.com/labs/12?returnTo=paths

import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x_batch, y_batch, lr):
    # Step 1: Zero all gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # Step 2: Forward pass
    logits = model(x_batch)

    # Step 3: Compute loss
    loss = F.cross_entropy(logits, y_batch)

    # Step 4: Backward pass
    loss.backward()

    # Step 5: Update parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # Step 6: Return loss as Python float
    return loss.item()


