# input layer: 10 inputs
# 1st hidden layer: 6 nodes and 1 bias unit (edges represent weight connections)
# 2nd hidden layer has 4 nodes and a node repr bias units
# output layer: 3 outputs

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(NeuralNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    
    def forward(self, x):
        return self.layers(x) # logits
    
model = NeuralNetwork(50, 3)
# print(model)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable model parameters: {num_parameters}")
 
# print(model.layers[0].weight)
# print(model.layers[0].weight.shape) # 30, 50
# print(model.layers[0].bias.shape) # 30

    

torch.manual_seed(123)
# model = NeuralNetwork(50, 3)
# print(model.layers[0].weight)

x = torch.rand((1, 50)) # our network expects 50-dimensional feature vectors
out = model(x)
print(out) # 3 outputs

# disable gd, use for inference
with torch.no_grad():
    out=model(x)
print(out)

# class prob
with torch.no_grad():
    out=torch.softmax(model(x), dim=1)
print(out)

# 5 training examples with two features each, 3 classes belong to 0, 2 belong to class 1, also make test set of two entries.
x_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
x_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

# toy dataset
class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(x_train, y_train)
test_ds = ToyDataset(x_test, y_test)

# purpose: use it to instantiate dataloader 
# print(len(train_ds)) # 5

torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}: ", x, y)


'''
Note that we specified a batch size of 2 above, but the 3rd batch only contains a single example. That’s because we have five training examples, which is not evenly divisible by 2. In practice, having a substantially smaller batch as the last batch in a training epoch can disturb the convergence during training. To prevent this, it’s recommended to set drop_last=True
which will drop the last batch in each epoch: drop_last=True

Batch 1: tensor([[-1.2000,  3.1000],
        [-0.5000,  2.6000]]) tensor([0, 0])
Batch 2: tensor([[ 2.3000, -1.1000],
        [-0.9000,  2.9000]]) tensor([1, 0])
'''


# Training loop
torch.manual_seed(123)
model = NeuralNetwork(2, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        outputs = model(x_train)

    torch.set_printoptions(sci_mode=False)
    prob = torch.softmax(outputs, dim=1)
    print(prob)
    '''
     means that the training example has a 99.91% probability of belonging to class 0 and a 0.09% probability of belonging to class 1. (The set_printoptions call is used here to make the outputs more legible.)
    '''
    # convert these into class label predictions using argmax (returns index posn of highest val in each row if wet dim=1 and highest value in each column if dim=0)
    predictions = torch.argmax(prob, dim=1)
    print(predictions) # [0,0,0,1,1] our desired output
    # verifying
    print(f"Number of correct predictions out of 5: {torch.sum(predictions==y_train)}")

    # print(outputs)

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct= 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = (labels == predictions)
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader)) # 1.0 since all are correct predictions

# saving the model
torch.save(model.state_dict(), 'model.pth')


# loading the model
model = NeuralNetwork(2, 3)# needs to match orig model exactly
print(model.load_state_dict(torch.load('model.pth'))) # <All keys matched>

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

print(tensor_1 + tensor_2)
# using cuda
# tensor_1 = tensor_1.to('cuda')
# tensor_2 = tensor_2.to('cuda')
# print(tensor_1 + tensor_2)

# device = torch.device("cuda")
# New: Transfer the model onto the GPU. 
# model.to(device)
# features, labels = features.to(device), labels.to(device)    

# better method: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

