# https://www.deep-ml.com/labs/2
import torch, torch.nn as nn

def build_model() -> nn.Module:
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()
             self.features = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 1->4
                nn.ReLU(),
                nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 4->8
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1) # (B, 8, 1, 1)
             )
             self.classifier = nn.Linear(8, 10) # 10 outputs

         def forward(self, x):
             x = self.features(x)
             x = x.view(x.size(0), -1)
             return self.classifier(x)
        
    model = TinyNet()
    params = sum(p.numel() for p in model.parameters())
    assert params <= 2048, params

    return model


'''
Linear neuron
    Needs 784 weights
    Pattern works at only one absolute position

Conv filter
    Needs 9 weights
    Pattern works everywhere

So conv says:
“If a vertical edge exists anywhere, I care.”

Linear says:
“If a vertical edge exists specifically here, I care.”

classifier only sees what patterns are present, not where.

# Parameter count for linear vs cnn
Linear(784 → 128): 784*128 + 128 = 100,480
Linear(128 → 64):  128*64 + 64   = 8,256
Linear(64 → 10):   64*10 + 10    = 650
-------------------------------------
Total ≈ 109,386 params


Conv1: (1*4*3*3 + 4)   = 40
Conv2: (4*8*3*3 + 8)   = 296
Linear: (8*10 + 10)    = 90
-----------------------------
Total ≈ 426 params
'''
