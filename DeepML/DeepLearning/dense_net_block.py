# a dense block is a stack of layers where each layer is connected to every other layer in the block in a feed-forward fashion

'''
Notes
Each layer \(L\) receives input from all preceding layers \(0, 1, 2, ..., L-1\) within the block.
Each layer \(L\) also passes its own output as input to all subsequent layers \(L+1, L+2, ..., N\) within the block.
This creates a very "dense" network of connections, hence the name
'''

import numpy as np

def conv2d(x, kernel, padding=0):
    if padding > 0:
        # x_pad = np.pad(x, pad_width=0, mode='constant')
        x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        x_pad = x

    bs, h, w, in_channel = x_pad.shape
    kh, kw, k_in, out_channels = kernel.shape
    output_ht = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((bs, output_ht, output_w, out_channels))
    for batch in range(bs):
        for i in range(output_ht):
            for j in range(output_w):
                for c_out in range(out_channels):
                    summ = 0.0
                    for c_in in range(in_channel):
                        summ += np.sum(x_pad[batch, i : i + kh, j : j + kw, c_in] * kernel[:, :, c_in, c_out])
                    output[batch, i, j, c_out] = summ
    return output


def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    kh, kw = kernel_size
    padding = (kh - 1) // 2
    ans = input_data.copy()
    for layer in range(num_layers):
        act = np.maximum(ans, 0)
        conv = conv2d(act, kernels[layer], padding=padding)
        ans = np.concatenate([ans, conv], axis=3)
    return ans

def main():
    X = np.random.randn(1, 2, 2, 1); kernels = [np.random.randn(3, 3, 2 + i*1, 1) * 0.01 for i in range(2)]; print(dense_net_block(X, 2, 1, kernels))
main()