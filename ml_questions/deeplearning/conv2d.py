# implementation of convolution layer with numpy and pytorch
"""
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)

Output: [[ 1.  1. -4.],
        [ 9.  7. -4.],
        [ 0. 14. 16.]]
"""

# Pytorch Implementation
import torch
import torch.nn as nn
import numpy as np


def simple_conv2d_pytorch(input_matrix, kernel, padding, stride):
    # in_channel = torch tensor of input_matrix, out_channel = tensor of kernel
    # Input Shape: (Batch Size, Channels, Height, Width) (assume bs, channels = 1, 1)
    input_tensor = (
        torch.tensor(input_matrix).float().unsqueeze(0).unsqueeze(0)
    )  # add batch size and channels

    kernel_tensor = torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0)
    out_channels, in_channels, kernel_height, kernel_width = kernel_tensor.shape
    output = nn.Conv2d(
        in_channels, out_channels, kernel_size=2, stride=stride, padding=padding
    )
    output.weight = nn.Parameter(kernel_tensor)
    ans = torch.round(output(input_tensor))
    return ans.squeeze(0).squeeze(0).detach().numpy()


# Numpy Implementation
"""
The height and width of the output matrix are calculated using the following formulas:
output_height=(input_height, padded - kernel_height) // stride + 1
output_width=(input_width, padded - kernel_width) // stride + 1
"""


def simple_conv2d(input_matrix, kernel, padding, stride):
    input_height, input_width = input_matrix.shape  # (4, 4)
    kernel_height, kernel_width = kernel.shape  # (2, 2)
    # calculate output height and width
    padded_input = np.pad(input_matrix, pad_width=padding, mode='constant')

    input_height_padded, input_width_padded = padded_input.shape
    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1

    # output shape = 3, 3
    output = np.zeros((output_height, output_width))

    # matrix multiply of kernel to input (basic math)
    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
            ]
            output[i, j] = np.sum(region * kernel)
    return output


def main():
    input_matrix = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    kernel = np.array([[1, 0], [-1, 1]])

    padding = 1
    stride = 2

    print("With Pytorch")
    print(simple_conv2d_pytorch(input_matrix, kernel, padding, stride))
    print("\n With Numpy")
    print(simple_conv2d(input_matrix, kernel, padding, stride))


main()
