import numpy as np
import torch

from algorithms.deeplearning.batch_normalization import batch_normalization


class TestResNet:
    def test_residual_block_forward_shape(self):
        """Test that a single ResidualBlock preserves spatial dimensions."""
        from papers.resnet.resnet34_scratch import ResidualBlock

        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_residual_block_downsample(self):
        """Test ResidualBlock with stride=2 halves spatial dims and changes channels."""
        from papers.resnet.resnet34_scratch import ResidualBlock
        import torch.nn as nn

        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
        )
        block = ResidualBlock(64, 128, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_resnet_forward_shape(self):
        """Test full ResNet34 produces correct output shape."""
        from papers.resnet.resnet34_scratch import ResNet, ResidualBlock

        model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)


class TestMNISTNetwork:
    def test_forward_shape(self):
        """Test that MNIST network produces correct output dimensions."""
        from models.mnist.from_scratch.network import Network

        net = Network([784, 30, 10])
        x = np.random.randn(784, 1)
        out = net.forward(x)
        assert out.shape == (10, 1)

    def test_sigmoid_range(self):
        """Test sigmoid output is in (0, 1)."""
        from models.mnist.from_scratch.network import Network

        net = Network([2, 2])
        vals = np.array([-100, -1, 0, 1, 100])
        for v in vals:
            s = net.sigmoid(v)
            assert 0 < s < 1 or np.isclose(s, 0) or np.isclose(s, 1)


class TestBatchNormalization:
    def test_output_shape(self):
        B, C, H, W = 4, 3, 8, 8
        X = np.random.randn(B, C, H, W)
        gamma = np.ones(C).reshape(1, C, 1, 1)
        beta = np.zeros(C).reshape(1, C, 1, 1)
        out = batch_normalization(X, gamma, beta)
        assert out.shape == (B, C, H, W)

    def test_zero_mean_unit_var(self):
        """With gamma=1 and beta=0, output should have ~zero mean and ~unit variance per channel."""
        np.random.seed(42)
        B, C, H, W = 8, 3, 16, 16
        X = np.random.randn(B, C, H, W) * 5 + 10  # offset data
        gamma = np.ones(C).reshape(1, C, 1, 1)
        beta = np.zeros(C).reshape(1, C, 1, 1)
        out = batch_normalization(X, gamma, beta)

        for c in range(C):
            channel_data = out[:, c, :, :]
            assert abs(np.mean(channel_data)) < 0.1, f"Channel {c} mean not near zero"
            assert abs(np.var(channel_data) - 1.0) < 0.2, f"Channel {c} variance not near 1"
