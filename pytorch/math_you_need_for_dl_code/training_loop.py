'''
Simple training loop in python
'''
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

# toy dataset
x = torch.randn(1000, 10)
y = (x.sum(dim=1) > 0).long() # labels: 0 or 1
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(10,64),  # 10 inp -> 64 op in hidden layer
    nn.ReLU(),  # apply activation
    nn.Linear(64,2) # takes the 64 inp from hidden layer -> 2 op (binary)
    )

# opt = optim.SGD(model.parameters(), lr=0.1)
opt = optim.Adam(model.parameters(), lr=1e-3)
# MSELoss is for regression, not classification.
loss_fn = nn.CrossEntropyLoss()

epoch_loss = 0.0
for epoch in range(10):
    model.train()
    for inp_x, target_y in dataloader:
        prediction = model(inp_x)
        loss = loss_fn(prediction, target_y)
        epoch_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # print(f"epoch {epoch} loss {loss.item():.4f}")
    epoch_loss /= len(dataloader)
    print(f"epoch {epoch} loss {epoch_loss:.4f}")


'''
Running through the output:
epoch 0 loss 0.5707
epoch 1 loss 0.5743
epoch 2 loss 0.4896
epoch 3 loss 0.4645
epoch 4 loss 0.3125
epoch 5 loss 0.2723
epoch 6 loss 0.2915
epoch 7 loss 0.1868
epoch 8 loss 0.2041
epoch 9 loss 0.2137

You can clearly see the loss decreasing,

Interesting question, what happens if we use SGD Loss + CrossEntropy isntead of Adam Loss
epoch 0 loss 0.5302
epoch 1 loss 0.4161
epoch 2 loss 0.2929
epoch 3 loss 0.3087
epoch 4 loss 0.2321
epoch 5 loss 0.1538
epoch 6 loss 0.1620
epoch 7 loss 0.1037
epoch 8 loss 0.1914
epoch 9 loss 0.1038

SGD Update:
- Same learning rate for all parameters
- Same scale every step
- Sensitive to gradient noise
- More oscillation

That explains why SGD loss: Jumps more + sometimes increases temporarily + still converges

Adam:
Adam:
- First moment (mean of gradients)
- Second moment (variance of gradients)
- Converges faster early
- Appears “smoother”
- Handles poorly scaled problems better

Our problem is simple (binary classification, shallow network, less epochs) hence SGD works better, nothing fancy here.
'''