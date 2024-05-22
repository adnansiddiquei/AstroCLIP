import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(*self.shape)


class Standardize(nn.Module):
    def __init__(self, return_mean_and_std: bool = True):
        super(Standardize, self).__init__()

        self.return_mean_and_std = return_mean_and_std

    def forward(self, x):
        # Compute mean and std along the spectrum length dimension
        eps = 1e-6  # for numerical stability, to avoid division by zero
        means = x.mean(dim=-1) + eps  # shape (batch_size, n_channels)
        stds = x.std(dim=-1) + eps  # shape (batch_size, n_channels)

        # Standardize the spectrum
        standardized_x = (x - means) / stds

        # Concatenate along the channel dimension
        if self.return_mean_and_std:
            return standardized_x, means, stds
        else:
            return standardized_x


class ExtractKey(nn.Module):
    def __init__(self, key):
        super(ExtractKey, self).__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]


class InterpolationImputeNans1d(nn.Module):
    def __init__(self):
        """
        Impute NaNs in a 1D tensor by linear interpolation with the left and right neighbors.

        The kernel used for convolution is [0.5, 0, 0.5], this is applied to the input tensor which should be of shape
        (batch_size, channel, length).
        """
        super(InterpolationImputeNans1d, self).__init__()
        self.kernel = torch.tensor([[[0.5, 0, 0.5]]], dtype=torch.float32)

    def forward(self, x):
        nans = torch.isnan(x)  # identify NaNs in the tensor
        x = torch.nan_to_num(x, nan=0.0)  # temporarily replace NaNs with zeros

        # Apply convolution to get the average of the left and right neighbors
        neighbors_avg = F.conv1d(x, self.kernel.to(x.device), padding=1)

        # Replace NaNs with the average of their neighbors
        x_imputed = torch.where(nans, neighbors_avg, x)

        return x_imputed
