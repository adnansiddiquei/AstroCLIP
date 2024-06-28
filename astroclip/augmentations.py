"""
This module contains PyTorch nn.Module subclasses that apply various augmentations to input Tensors.

The augmentations have been written to be as general as possible. However, they are generally intended
for batches of spectra of shape (batch_size, 1, 7781) and batches of images of shape (batch_size, 3, 256, 256).
"""

import torch
import torch.nn as nn


class Roll(nn.Module):
    def __init__(self):
        """
        Roll the tensor along the last two dimensions by a random amount between [-5, 5].
        """
        super(Roll, self).__init__()

    def forward(self, x: torch.Tensor):
        shifts = torch.randint(low=-5, high=5, size=(2,), device=x.device).tolist()

        return x.roll(shifts, (-1, -2))


class AddGaussianNoise(nn.Module):
    def __init__(self, mean: float = 0, std: float = 1):
        """
        Add Gaussian noise to the tensor.

        Parameters
        ----------
        mean : float
            The mean of the Gaussian distribution.
        std : float
            The standard deviation of the Gaussian distribution.
        """
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        return x + (torch.randn_like(x, device=x.device) * self.std) + self.mean


class SpectrumNoising(nn.Module):
    def __init__(
        self, observed_spectra_std_dev: torch.Tensor, noise_strength: float = 0.2
    ):
        """
        Add Gaussian noise to the observed spectra.

        Expects input spectra to be of shape (batch_size, 1, 7781).

        Parameters
        ----------
        observed_spectra_std_dev : torch.Tensor
            The standard deviation of the observed spectra. This is the tensor computed and output by the
            compute_observed_spectra_std_dev.py script.
            See Section (3.2) of the report for more details on what this is.
        noise_strength : float
            A constant scaling parameter for how much noise to add to the observed spectra.
            See Section (3.2) of the report for more details on what this is.
        """
        super(SpectrumNoising, self).__init__()
        self.observed_spectra_std_dev = observed_spectra_std_dev
        self.noise_strength = noise_strength

    def forward(self, spectrum):
        noise = (
            # create standard normal gaussian noise
            torch.randn_like(spectrum).to(spectrum.device)
            # then scale by spectral standard deviations, see Section (3.2) of report
            * self.observed_spectra_std_dev.expand_as(spectrum).to(spectrum.device)
            # then scale by noise_strength hyperparameter
            * self.noise_strength
        )

        noisy_spectrum = spectrum + noise

        return noisy_spectrum
