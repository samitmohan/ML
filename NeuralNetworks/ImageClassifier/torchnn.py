# Process -:
# Downloads the MNIST dataset, converts it into PyTorch tensors, and creates a data loader. The data loader loads the dataset in batches of 32 images.
# The convolutional layers are designed for processing 28x28 images with a single channel (grayscale). The network aims to classify images into one of 10 classes (digits 0-9).
# Within each epoch, it iterates over the batches of data, computes the model's predictions, calculates the loss, performs backpropagation to update the model's parameters, and prints the loss for each epoch.

import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor  # convert images > tensors

# MNIST : 0-9 class data for numbers.

# Get Data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

dataset = DataLoader(train, 32)  # pass train partition, batches of 32 images

# Prediction : now our clf is trained, need to apply
def predict():
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))  # load weights in classifier
        img_2 = Image.open("img_1.jpg")
        img_0 = Image.open("img_2.jpg")
        img_9 = Image.open("img_3.jpg")
        # convert to tensor
        img_tensor_2 = (
            ToTensor()(img_2).unsqueeze(0).to("cpu")
        )  # unsqueeze : add batch dimension bcs neural network expect input data in batch format

        img_tensor_0 = ToTensor()(img_0).unsqueeze(0).to("cpu")
        img_tensor_9 = ToTensor()(img_9).unsqueeze(0).to("cpu")

        print("Predictions : (Should be 2, 0, 9)")
        print(torch.argmax(clf(img_tensor_2)))  # prediciton
        print(torch.argmax(clf(img_tensor_0)))
        print(torch.argmax(clf(img_tensor_9)))


# Images from MNIST : 1,28,28 Shape. Classes = 0-9 (10)
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # model layers
            nn.Conv2d(1, 32, (3, 3)),  # 1 : input channel (b/w image), filters, shape
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  # 32 : input channels, filters, shape
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),  # 64 : input channels, filters, shape
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)  # apply model for FP


# Instance, loss, optimizer
clf = ImageClassifier().to("cpu")
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()  # calculates loss function (minimise later)

# Training

if __name__ == "__main__":
    # Comment for prediction
    for epoch in range(10):  # 10 iterations for training
        for batch in dataset:  # every batch, unpack that batch (x,y)
            x, y = batch
            x, y = x.to("cpu"), y.to("cpu")
            y_pred = clf(x)
            loss = loss_fn(y_pred, y)

            # Apply backpropagation
            opt.zero_grad()  # reset gradient of all parameters before computing new gradients during back pass
            loss.backward()
            opt.step()  # update weights and biases of model parameters after backpropagation

        print(f"Epoch {epoch} loss is {loss.item()}")  # item : loss value

    # Save model to environment
    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)

    predict()
