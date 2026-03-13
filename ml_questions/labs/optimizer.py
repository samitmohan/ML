# https://www.deep-ml.com/labs/3

import math
import torch
from torch.optim.optimizer import Optimizer

'''
MiniSGD + Momentum
- SGD zig-zags, momentum remembers past gradients
Update rule:
    - learning_rate = some factor * learning_ratet-1 + (learning gradient)t  
    - theta(t) = theta(t-1) - n * learning_rate
'''

class MyOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                # implementation here
                grad = p.grad
                state = self.state[p]
                if 'velocity' not in state:
                    state['vecocity'] = torch.zeros_like(p)
                v = state['velocity']

                v.mul_(momentum).add_(grad)
                p.add(v, alpha=-lr)

        return loss
