"""
This module contains PyTorch nn.Module subclasses that apply various transforms to input Tensors.

The transforms have been written to be as general as possible. However, they are generally intended
for batches of spectra of shape (batch_size, 1, 7781) and batches of images of shape (batch_size, 3, 256, 256).

Some of the transforms are very dataset specific, such as DropInvalidSpectra and DropOnRedshift.
"""

import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, dims):
        """
        Permute the dimensions of a tensor.

        Parameters
        ----------
        dims : list
            A list of integers representing the new order of the dimensions.
        """
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    def __init__(self, dim):
        """
        Squeeze the tensor along a given dimension.

        Parameters
        ----------
        dim : int
            The dimension to squeeze.
        """
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(self.dim)


class DropInvalidSpectra(nn.Module):
    def __init__(self):
        """
        Drop the spectrum, image, targetid, and redshift of samples where the spectrum is all zeros.
        """
        super(DropInvalidSpectra, self).__init__()

    def forward(self, batch):
        spectrum = batch['spectrum']
        mask = (spectrum.abs().sum(dim=1) != 0).squeeze()

        batch['spectrum'] = spectrum[mask]
        batch['image'] = batch['image'][mask]
        batch['targetid'] = batch['targetid'][mask]
        batch['redshift'] = batch['redshift'][mask]

        return batch


class DropOnRedshift(nn.Module):
    def __init__(self, z_min=0.0, z_max=0.8):
        """
        Drop the spectrum, image, targetid, and redshift of samples where the redshift is not within the desired range.

        Parameters
        ----------
        z_min : float
            Minimum redshift to keep.
        z_max : float
            Maximum redshift to keep.
        """
        super(DropOnRedshift, self).__init__()
        self.z_min = z_min
        self.z_max = z_max

    def forward(self, batch):
        redshift = batch['redshift']

        # Create a mask for redshifts within the desired range
        mask = (redshift <= self.z_max) & (redshift >= self.z_min)

        # Apply mask to filter the batch along the batch dimension
        batch['image'] = batch['image'][mask]
        batch['spectrum'] = batch['spectrum'][mask]
        batch['targetid'] = batch['targetid'][mask]
        batch['redshift'] = batch['redshift'][mask]

        return batch


class Standardise(nn.Module):
    def __init__(self, return_mean_and_std: bool = True):
        """
        Standardise a tensor along the last dimension.

        Parameters
        ----------
        return_mean_and_std : bool
            If True, return the mean and standard deviation of the input tensor along the last dimension.
        """
        super(Standardise, self).__init__()

        self.return_mean_and_std = return_mean_and_std

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor]:
        # Compute mean and std along the spectrum length dimension
        eps = 1e-6  # for numerical stability, to avoid division by zero
        means = x.mean(dim=-1, keepdims=True) + eps  # shape (batch_size, n_channels)
        stds = x.std(dim=-1, keepdims=True) + eps  # shape (batch_size, n_channels)

        # Standardise the spectrum
        standardized_x = (x - means) / stds

        # Concatenate along the channel dimension
        if self.return_mean_and_std:
            return standardized_x, means, stds
        else:
            return standardized_x


class StandardiseAlong(nn.Module):
    def __init__(self, dims: list | int):
        """
        Standardise a tensor along the given dimensions.

        Parameters
        ----------
        dims : list | int
            A list of dimensions along which to standardise the tensor.
        """
        super(StandardiseAlong, self).__init__()
        self.dims = dims

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor]:
        mean = x.mean(dim=self.dims, keepdim=True)
        std = x.std(dim=self.dims, keepdim=True)

        std = torch.where(std == 0, std + 1e-10, std)

        normalised = (x - mean) / std

        return normalised


class ExtractKey(nn.Module):
    def __init__(self, key):
        """
        Extract a key from a dictionary.

        Parameters
        ----------
        key : str
            The key to extract from the dictionary.
        """
        super(ExtractKey, self).__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]
