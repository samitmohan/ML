'''
Resnet50 for ImageNet + Support for basicblock(for resnet18,34)
Layers : [3,4,6,3]
'''
import os
from PIL import Image
import torch
from torch.optim import SGD
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock:
    ''' for resnet18, 34 , only supports group=1, widtth=64'''
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, groups=1, base_width=64) -> None:
        assert groups==1 and base_width==64
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = []
        if stride != 1 or in_channels != self.expansion*channels:
            self.downsample = [
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            ]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out += x(self.downsample)
        out = out.relu()
        return out

# Bottleneck
class BottleneckBlock(nn.Module):
    expansion = 4 # standard for resnet50
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # to reduce channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), # main processing 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False), # expand channels (back to)
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# Resnet Class
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64 # keep track of number of channels expected by next layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # standard for CNN
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output feature will have 64 channels
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Data Loader for Imagenet
IMAGENET_DATA = ''
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(IMAGENET_DATA, 'train'), transform=train_transform)
valid_dataset = datasets.ImageFolder(os.path.join(IMAGENET_DATA, 'val'), transform=valid_transform)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Training
num_classes = 1000 
epochs = 20
learning_rate = 0.01

model = ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes).to(device)
model_save_path = "resnet50_imagenet_scratch.pth"
if os.path.exists(model_save_path):
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded pre-trained model from {model_save_path}")
    except Exception as e:
        print(f"Error loading model: {e}. Training from scratch.")

loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.9)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct_train, total_train = 0,0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)
            # forward
            outputs = model(images)
            loss = loss_fn(outputs)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()


    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # testing
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

# classify single image
image_path = "sultan.jpeg"
if os.path.exists(image_path):
    try:
        with open("imagenet_classes.txt") as f:
            class_labels = [s.strip() for s in f.readlines()]
    except FileNotFoundError:
        print("Image Net Classifier File Not Found.")
        class_labels = [f"Class_{i}" for i in range(num_classes)]
    
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    try:
        image = Image.open(image_path)
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()

        predicted_label = class_labels[pred_class]
        print(f"This image is classifed as: {predicted_label}")
        top5_prob, top5_catid = torch.topk(prob, 5)
        print("Top 5 predictions:")
        for i in range(top5_prob.size(0)):
            print(f"- {class_labels[top5_catid[i]]}: {top5_prob[i].item():.4f}")


    except FileNotFoundError:
            print(f"\nSample image '{image_path}' not found. Skipping single image classification.")
    except Exception as e:
        print(f"\nAn error occurred during single image classification: {e}")
else:
    print(f"\nSample image '{image_path}' not found. Skipping single image classification.")