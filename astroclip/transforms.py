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


class StandardizeAndAugment(nn.Module):
    def __init__(self):
        """
        Standardize the spectrum and augment it with the mean and standard deviation.

        The input tensor should have shape (batch_size, 1, spectrum_length). The output tensor will have shape
        (batch_size, 3, spectrum_length) where the first channel is the standardized spectrum, the second channel is
        the mean of the spectrum, and the third channel is the standard deviation of the spectrum.
        """
        super(StandardizeAndAugment, self).__init__()

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
        augmented_x = torch.cat((standardized_x, mean_channel, std_channel), dim=1)

        return augmented_x
