# https://www.deep-ml.com/labs/1
import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        brightness_shift = torch.randn_like(x) * 0.05
        contrast_shift = torch.randn_like(x) * 0.05
        noise = torch.randn_like(x) * 0.05
        # two transformations -> ToTensor() and Normalise
        x = (x - torch.mean(x)) / torch.std(x)
        x = x + contrast_shift + brightness_shift
        x = x.clamp(min=-3.0, max=3.0)
        x = x + noise

        return x

