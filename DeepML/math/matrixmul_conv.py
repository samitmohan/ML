# Matrix multiplication is just convolution with 1x1 kernel and stride 1.
# Input Tensor : B, Kernel Tensor : A
import numpy as np
import torch 
import torch.nn.functional as F

def matrix_mul(A, B):
    '''
    A : numpy arr of shape M * K
    B : numpy arr of shape K * N
    output C = shape = M * N (Matrix Multiply)
    '''
    M, K = A.shape
    K_b, N = B.shape
    if K != K_b:
        raise ValueError("Incompatible shapes")
    
    input = torch.from_numpy(B).float().unsqueeze(0).unsqueeze(0) # add batch and channel dim (1, 1, K, N)
    kernel = torch.from_numpy(A).float().unsqueeze(1).unsqueeze(3) # add in_channels and kernel_width (M, 1, K, 1)

    matrix_mul = F.conv2d(input, kernel, stride=1, padding=0)
    C = matrix_mul.squeeze(0).squeeze(1) # remove batch dim and kernel_width dim, becomes M * N rather than (1, M, 1, N) -> squeeze(0) -> (M, 1, N) -> squeeze(1) -. (M, N) what we want
    return C.numpy()


def main():
    A = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.float32) # Shape (2, 3) -> M=2, K=3

    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=np.float32) # Shape (3, 2) -> K=3, N=2

    expected_C = np.array([[58, 64],
                            [139, 154]], dtype=np.float32)
    result_C = matrix_mul(A, B)
    try:
        assert np.allclose(result_C, expected_C)
        print("Assertion passed: The conv2d result matches the expected result.")
    except AssertionError:
        print("Assertion Failed: Conv2d Result does not match expected result!")

    matrix_mul(A, B)
main()