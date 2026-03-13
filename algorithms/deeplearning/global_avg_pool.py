# https://www.deep-ml.com/problems/114

"""
Implement a function that performs Global Average Pooling on a 3D NumPy array representing feature maps from a convolutional layer.
The function should take an input of shape (height, width, channels) and return a 1D array of shape (channels,), where each element is the average of all values in the corresponding feature map.

Input:

x = np.array([[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]])

Output:

[5.5, 6.5, 7.5]

Reasoning:

For each channel, compute the average of all elements. For channel 0: (1+4+7+10)/4 = 5.5, for channel 1: (2+5+8+11)/4 = 6.5, for channel 2: (3+6+9+12)/4 = 7.5.

This operation effectively summarizes each feature map into a single value, capturing the essence of the features learned by the network.
GAP computes the average of each entire feature map, resulting in a single value per channel.

GAP(x)channel = (1 / H * W)  * sum (i=1) * sum (j=1) xi,j,channel

Bsaically mean over spatial dims (ht and width) for each channel
Global Average Pooling is a key component in architectures like ResNet, where it is used before the final classification layer. It allows the network to handle inputs of varying sizes, as the output depends only on the number of channels, not the spatial dimensions.
"""

import numpy as np


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    height, width, channels = x.shape
    length = height * width
    ans_sums = [0.0] * channels
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                ans_sums[k] += x[i, j, k]

    ans = [a / length for a in ans_sums]
    return [float(x) for x in ans]

    # OR just use np.mean
    # ans = np.mean(x, axis=(0,1)) # 0 is height, 1 is width
    # return ans


def main():
    print(
        global_avg_pool(x=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    )


main()


"""
In Pytorch: 
input_tensor = input_matrix.unsqueeze(0).unsqueeze(0)
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
print(maxpool(input_tensor))
gap = nn.AdaptiveMaxPool2d((1,1))
ans = gap(input_tensor).squeeze(0).squeeze(0)
print(ans)
"""
