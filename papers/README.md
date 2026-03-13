# Paper Implementations

## ResNet - Deep Residual Learning for Image Recognition

**He, Zhang, Ren, Sun (2015)** - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

Three implementations of the ResNet architecture:

| File | Approach |
|------|----------|
| [`resnet/resnet34_scratch.py`](resnet/resnet34_scratch.py) | ResNet-34 built from `nn.Conv2d` and `nn.BatchNorm2d`, trained on CIFAR-10 |
| [`resnet/resnet_pytorch.py`](resnet/resnet_pytorch.py) | Using `torchvision.models.resnet50` pretrained on ImageNet |
| [`resnet/resnet_huggingface.ipynb`](resnet/resnet_huggingface.ipynb) | HuggingFace `transformers` pipeline for image classification |

Key concepts implemented: residual blocks with skip connections, downsampling via stride-2 convolutions, adaptive average pooling, batch normalization.

## Perceptron

**Rosenblatt (1958)** - "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

[`perceptron/perceptron.py`](perceptron/perceptron.py) - Implementation of the original perceptron learning algorithm. Includes the original 1958 paper PDF.
