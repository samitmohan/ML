'''
ResNet-34 with CIFAR-10 images
Blocks : [3,3,6,3]
'''
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD
from tqdm import tqdm  
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]
    )

    if test:
        dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform,)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    
    # load dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

train_loader, valid_loader = data_loader(data_dir='./data', batch_size=64)
test_loader = data_loader(data_dir='./data', batch_size=64, test=True)


class ResidualBlock(nn.Module):
    ''' One Residual Block : forward pass '''

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        )

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    # forward pass
    def forward(self, x):
        residual = x # input
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), # out_channels
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4 layers : 3, 3, 6, 3 residual blocks
        
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            # need to downsample to make output size same as input
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels), # out_channels
            )
        
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample)) # call residual block
        self.in_channels = channels # update after first block
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


num_classes = 10
epochs = 20
batch_size = 32
learning_rate = 0.01

model = ResNet(ResidualBlock, [3,4,6,3]).to(device)
model_save_path = "resnet_scratch.pth"
if os.path.exists(model_save_path):
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded pre-trained model from {model_save_path}")
    except Exception as e:
        print(f"Error loading model: {e}. Training from scratch.")

loss_fn = nn.CrossEntropyLoss()
optimiser = SGD(model.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.9)
total_step = len(train_loader)



# Load images in batches using train_loader for every epoch
# Predict on labels and calculate loss b/w predictions and ground truth using loss fn
# loss.backward(), update weights, optimiser.step() 
# Test model on validation set.
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct_train, total_train = 0, 0

    # Use tqdm for progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)
            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update progress bar
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            t.set_postfix(loss=loss.item())

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        correct_valid, total_valid = 0, 0
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid
        print(f"Validation Accuracy: {valid_accuracy:.2f}%")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved")

image_path = "sultan.jpeg" 
if os.path.exists(image_path):
    try:
        # CIFAR-10 class labels
        class_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # Preprocessing for classification (should match training transforms for the model)
        preprocess_single = transforms.Compose([
            transforms.Resize((224, 224)), # Keep consistent with training resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess_single(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()

        predicted_label = class_labels[pred_class]
        print(f"\nClassifying '{image_path}':")
        print(f"The image is classified as: {predicted_label}")
        
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        print("Top 5 predictions:")
        for i in range(top5_prob.size(0)):
            print(f"- {class_labels[top5_catid[i]]}: {top5_prob[i].item():.4f}")

    except FileNotFoundError:
        print(f"\nSample image '{image_path}' not found. Skipping single image classification.")
    except Exception as e:
        print(f"\nAn error occurred during single image classification: {e}")
else:
    print(f"\nSample image '{image_path}' not found. Skipping single image classification.")
    print("Please ensure you have an image like 'sultan.jpeg' or similar, or update the image_path.")