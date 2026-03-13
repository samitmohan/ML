# https://www.deep-ml.com/problems/115
"""
Implement a function that performs Batch Normalization on a 4D NumPy array representing a batch of feature maps in the BCHW format (batch, channels, height, width).
The function should normalize the input across the batch and spatial dimensions for each channel, then apply scale (gamma) and shift (beta) parameters.
Use the provided epsilon value to ensure numerical stability.

Notes

BN is a widely used technique that helps to accelerate the training of deep neural networks and improve model performance.
By normalizing the inputs to each layer so that they have a mean of zero and a variance of one, BN stabilizes the learning process, speeds up convergence, and introduces regularization,
which can reduce the need for other forms of regularization like dropout.

Goal : reducing internal covariate shift which occurs when the distribution of inputs to a layer changes during training as the model weights get updated

Steps:
- Compute the Mean and Variance: For each mini-batch, compute the mean and variance of the activations for each feature (dimension).
- Normalize the Inputs: Normalize the activations using the computed mean and variance.
- Apply Scale and Shift: After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.
- Training and Inference: During training, the mean and variance are computed from the current mini-batch. During inference, a running average of the statistics from the training phase is used.



"""

import numpy as np


def batch_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # Mean over (B, H, W)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)
    # Normalize X using (X - mean) / sqrt(variance + epsilon), then scale by gamma and shift by beta.
    X_normalised = (X - mean) / np.sqrt(variance + epsilon)
    scale_shift = X_normalised * gamma + beta
    return scale_shift


def main():
    B, C, H, W = 2, 2, 2, 2
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)
    print(batch_normalization(X, gamma, beta))


main()

"""
X is
[[[[-0.34743979 -0.05404846]
   [-0.84480325  1.15382749]]

  [[ 0.24700145  1.29851457]
   [ 0.84627406 -1.09791293]]]


 [[[ 1.12067623  0.52700588]
   [-2.02446819 -1.65604761]]

  [[-0.8571145  -1.08871324]
   [ 0.62107179  1.49788034]]]]

   Can see there are 2x2 row col matrices, and 4 channels overall and 2 batches of them
"""
