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


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(self.dim)


class DropInvalidSpectra(nn.Module):
    def __init__(self):
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


class Standardize(nn.Module):
    def __init__(self, return_mean_and_std: bool = True):
        """
        Standardize a tensor along the last dimension.

        Parameters
        ----------
        return_mean_and_std : bool
            If True, return the mean and standard deviation of the input tensor along the last dimension.
        """
        super(Standardize, self).__init__()

        self.return_mean_and_std = return_mean_and_std

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor]:
        # Compute mean and std along the spectrum length dimension
        eps = 1e-6  # for numerical stability, to avoid division by zero
        means = x.mean(dim=-1, keepdims=True) + eps  # shape (batch_size, n_channels)
        stds = x.std(dim=-1, keepdims=True) + eps  # shape (batch_size, n_channels)

        # Standardize the spectrum
        standardized_x = (x - means) / stds

        # Concatenate along the channel dimension
        if self.return_mean_and_std:
            return standardized_x, means, stds
        else:
            return standardized_x


class NormaliseSpectrum(nn.Module):
    def __init__(
        self, restframe_wavelengths: torch.Tensor, normalisation_range=(5300, 5850)
    ):
        """
        Normalise the spectrum using the median flux over the wavelength range [5300, 5850] Angstroms, in the rest
        frame.
        """
        super(NormaliseSpectrum, self).__init__()
        self.register_buffer('restframe_wavelengths', restframe_wavelengths)
        self.normalisation_range = normalisation_range

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor]:
        assert x.shape[-1] == self.restframe_wavelengths.shape[-1], (
            f'Spectrum dimensions must be equal to wavelength '
            f'dimensions {len(self.restframe_wavelengths)}'
        )

        mask = (self.restframe_wavelengths >= self.normalisation_range[0]) & (
            self.restframe_wavelengths <= self.normalisation_range[1]
        )

        median_flux = x[:, :, mask].median(dim=-1, keepdim=True)[0]

        normalised_spectrum = x / (median_flux + 1e-6)

        return normalised_spectrum


class MeanNormalise(nn.Module):
    def __init__(self, dims):
        super(MeanNormalise, self).__init__()
        self.dims = dims

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor]:
        mean = x.mean(dim=self.dims, keepdim=True)
        std = x.std(dim=self.dims, keepdim=True)

        normalised = (x - mean) / std

        return normalised


class ExtractKey(nn.Module):
    def __init__(self, key):
        super(ExtractKey, self).__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]


class ToRGB(nn.Module):
    def __init__(self, scales=None, m=0.03, Q=20, bands=['g', 'r', 'z']):
        """
        Takes in image of shape (N, C, L, W) converts from native telescope image scaling to RGB.
        Taken directly from Stein, et al, (2021).
        """
        super(ToRGB, self).__init__()
        rgb_scales = {
            'u': (2, 1.5),
            'g': (2, 6.0),
            'r': (1, 3.4),
            'i': (0, 1.0),
            'z': (0, 2.2),
        }
        if scales is not None:
            rgb_scales.update(scales)

        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])

        # rearrange scales to correspond to image channels after swapping
        self.scales = torch.tensor([self.scales[i] for i in self.axes])

    def forward(self, image):
        self.scales = self.scales.to(image.device)
        image = image.permute(0, 2, 3, 1)

        I = torch.sum(torch.clamp(image * self.scales + self.m, min=0), dim=-1) / len(
            self.bands
        )

        fI = torch.arcsinh(self.Q * I) / torch.sqrt(torch.tensor(self.Q, dtype=I.dtype))
        I = I + (I == 0).float() * 1e-6

        image = image * (self.scales + self.m * (fI / I).unsqueeze(-1)).to(image.dtype)

        image = torch.clamp(image, 0, 1)

        return image.permute(0, 3, 1, 2)


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
