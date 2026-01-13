# https://www.deep-ml.com/problems/151?from=AlexNet

'''
During training, randomly turn off neurons so the network can’t depend on any single path.
Imagine a team solving problems:
- If the same few people always work → fragile team
- If random people are absent each day → everyone must learn

Randomly removes neurons and Forces the network to distribute knowledge
p = 0.5 → 50% of neurons are turned off
Each neuron is independently:
- dropped with probability p
- kept with probability 1 - p

Dropout is applied per forward pass

Generate a random mask (Same shape as input)
Each element is:
- 0 (drop)
- 1 (keep) / (1 - p) (scale up to maintain expected value)

Multiply input element-wise by the mask, Dropped neurons become 0
Multiply remaining values by 1 / (1 - p) to keep expected sum the same

Without scaling:
- Expected activation magnitude decreases
- Network behaves differently at inference

With scaling:
- Expected value stays the same
- Training and inference distributions match

During inference:
No randomness. No masking. Just return input as is. (All neurons are active)

during backward pass:
- Reuse the same dropout mask
- Gradients for dropped neurons → zero
- Gradients for kept neurons → scaled by same factor
'''

import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return x
                
        self.mask = np.random.binomial(1, 1 - self.p, x.shape) # create mask
        return x * self.mask / (1 - self.p) # apply
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask / (1 - self.p)