import urllib.request
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()  

filename = "sultan.jpeg"
input_img = Image.open(filename)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = preprocess(input_img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

with torch.no_grad():
    output = model(input_batch)

print("Raw logits:", output[0])

prob = F.softmax(output[0], dim=0)
with open("imagenet_classes.txt") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(prob, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())