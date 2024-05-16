import torch


class PermuteTransform:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x: torch.Tensor):
        return x.permute(*self.dims)


class ReshapeTransform:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x: torch.Tensor):
        return x.view(*self.shape)
