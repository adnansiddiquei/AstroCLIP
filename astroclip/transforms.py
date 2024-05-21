import torch
import torch.nn as nn


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
        """
        Standardize a batch.

        The input tensor should have shape (batch_size, 1, *dataset_dims). The output tensor will have shape
        (batch_size, 3, *dataset_dims) if return_mean_and_std is True, and (batch_size, 1, *dataset_dims) if
        return_mean_and_std is False.

        The first channel will be the standardized dataset, the second channel will be the mean of the dataset, and the
        third channel will be the standard deviation of the dataset.

        Parameters
        ----------
        return_mean_and_std : bool
            Whether to return the mean and standard deviation of the dataset along with the standardized dataset.
        """
        super(Standardize, self).__init__()

        self.return_mean_and_std = return_mean_and_std

    def forward(self, x):
        # Compute mean and std along the spectrum length dimension
        means = x.mean(dim=-1, keepdim=True)
        stds = x.std(dim=-1, keepdim=True)

        # Standardize the spectrum
        standardized_x = (x - means) / stds

        # Expand the mean and std to the same shape as x
        mean_channel = means.expand_as(x)
        std_channel = stds.expand_as(x)

        # Concatenate along the channel dimension
        if self.return_mean_and_std:
            return torch.cat((standardized_x, mean_channel, std_channel), dim=1)
        else:
            return standardized_x


class ExtractKey(nn.Module):
    def __init__(self, key):
        super(ExtractKey, self).__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]
