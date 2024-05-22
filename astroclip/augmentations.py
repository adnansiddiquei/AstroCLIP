import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_1d_gaussian_kernel


class Roll(nn.Module):
    def __init__(self, mu: float = 0, sigma: float = 1):
        """
        Roll the tensor along the last two dimensions by a random amount equal to a normal distribution with mean `mu`
        and standard deviation `sigma`. The amount of shift is rounded to the nearest integer.

        Parameters
        ----------
        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        """
        super(Roll, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        shifts = (
            torch.round((torch.randn(2, device=x.device) * self.sigma) + self.mu)
            .int()
            .tolist()
        )
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


class GaussianSmoothing1d(nn.Module):
    def __init__(self, sigma: float, kernel_size: int):
        """
        Creates and applies a 1D Gaussian smoothing filter to a tensor of shape (batch_size, channels, length).

        Parameters
        ----------
        kernel_size : int
            The size of the kernel. It should be an odd number.
        sigma : float
            The standard deviation of the Gaussian kernel.
            provided.
        """
        super(GaussianSmoothing1d, self).__init__()

        self.kernel_size = kernel_size
        self.sigma = sigma

        self.register_buffer('kernel', create_1d_gaussian_kernel(sigma, kernel_size))

    def forward(self, x: torch.Tensor):
        kernel = self.kernel.to(x.device)

        return F.conv1d(x, kernel.view(1, 1, -1), padding=self.kernel_size // 2)


class SpectrumNoising(nn.Module):
    def __init__(
        self, observed_spectra_std_dev: torch.Tensor, noise_strength: float = 0.2
    ):
        """
        Add Gaussian noise to the observed spectra.

        Parameters
        ----------
        observed_spectra_std_dev : torch.Tensor
            The standard deviation of the observed spectra. This is the tensor computed and output by the
            compute_observed_spectra_std_dev.py script.
        noise_strength : float
            The strength of the noise to be added. For each element in the observed spectra, the noise added will be
            sampled from a Gaussian distribution with mean 0 and standard deviation equal to the
            observed_spectra_std_dev * std dev of spectra * noise_strength.
        """
        super(SpectrumNoising, self).__init__()
        self.observed_spectra_std_dev = observed_spectra_std_dev
        self.noise_strength = noise_strength

    def forward(self, spectrum):
        stds = spectrum.std(dim=-1, keepdims=True).repeat(1, 1, spectrum.shape[-1])

        noise = (
            torch.randn_like(spectrum)  # create gaussian noise
            *
            # then scale noise by observed_spectra_std_dev, this ensures that wavelengths that naturally have more
            # variance will have more noise added to them
            self.observed_spectra_std_dev.expand_as(spectrum)
            *
            # then scale the noise by the individual std dev of each spectrum so that spectra which naturally have
            # more variance or larger scales will have more noise added
            stds
            * self.noise_strength  # scale noise by noise_strength parameter
        ).to(spectrum.device)

        noisy_spectrum = spectrum + noise

        return noisy_spectrum
