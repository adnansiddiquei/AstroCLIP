import torch.nn as nn
import torch
import numpy as np
import pytorch_lightning as L
import os
import datasets
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.axes import Axes
import yaml


def freeze_all_layers(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


# Function to unfreeze a specific layer by name
def unfreeze_layer_by_name(model: nn.Module, layer_name: str):
    # Check if the layer name exists in the model
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")


def set_trainable_layers(model: nn.Module, layer_names: list):
    # Freeze all layers
    freeze_all_layers(model)

    # Unfreeze the specified layers
    for layer_name in layer_names:
        unfreeze_layer_by_name(model, layer_name)


def create_1d_gaussian_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        The standard deviation of the Gaussian.
    kernel_size : int
        The size of the kernel. It should be an odd number.

    Returns
    -------
    torch.Tensor
        The 1D Gaussian kernel.
    """
    assert kernel_size % 2 == 1, 'Kernel size should be an odd number.'

    # Create a tensor of size 'size' with values from -size//2 to size//2
    x = torch.arange(
        -int(kernel_size / 2), int(kernel_size / 2) + 1, dtype=torch.float32
    )

    kernel = torch.exp(-(x**2) / (2 * sigma**2))  # compute the 1D Gaussian filter
    kernel = kernel / kernel.sum()  # normalise the kernel

    return kernel


def copy_weights(source: nn.Module, target: nn.Module) -> None:
    """
    Copy the weights from one model to another.

    Parameters
    ----------
    source : nn.Module
        The model to copy the weights from.
    target : nn.Module
        The model to copy the weights to.
    """
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)


def calc_model_parameter_count(model: nn.Module | L.LightningModule) -> int:
    """
    Calculate the number of parameters in a model
    """
    return np.sum([p.numel() for p in model.parameters()])


def create_dir_if_required(script_filepath: str, dir_name: str) -> str:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    script_filepath : str
        The path to the script calling this function.
    dir_name : str
        The name of the directory to create, relative to the path provided in the first function.

    Returns
    -------
    str
        The path to the directory created.
    """
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    dir_to_make = os.path.join(cwd, dir_name)

    if not os.path.exists(os.path.join(cwd, dir_name)):
        os.makedirs(dir_to_make)

    return dir_to_make


def download_desi_dataset(cache_dir: str) -> datasets.dataset_dict.DatasetDict:
    """
    Download the DESI dataset used in the AstroCLIP paper and save it in the cache directory.

    If the dataset is already downloaded, it will be loaded from the cache directory.

    Parameters
    ----------
    cache_dir : str
        The path to the cache directory where the dataset will be saved.

    Returns
    -------
    datasets.dataset_dict.DatasetDict
        The DESI dataset.
    """
    cwd = os.path.dirname(os.path.realpath(__file__))

    dataset_dir = f'{cache_dir}/datasets_astroclip'

    dataset = load_dataset(
        f'{cwd}/legacy_survey.py',
        cache_dir=dataset_dir,
        trust_remote_code=True,
    )

    dataset.set_format(type='torch')

    return dataset


class SpectralStdDevCalculator:
    def __init__(
        self,
        dataloader: DataLoader,
        transforms: nn.Sequential = None,
        augmentations: nn.Sequential = None,
    ):
        """
        Compute the standard deviation of the observed spectra.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader containing the observed spectra within each batch.
        transforms : nn.Sequential, optional
            A sequence of transformations to apply to the observed spectra. The default is None.
        augmentations : nn.Sequential, optional
            A sequence of augmentations to apply to the observed spectra. The default is None.

        Methods
        -------
        compute() -> torch.Tensor
            Compute the standard deviation of the observed spectra. It returns a tensor of shape (1, spectrum_length)
            where each element is the standard deviation of the observed spectra at that wavelength.
        """
        self.dataloader = dataloader
        self.transforms = transforms
        self.augmentations = augmentations
        self.std_dev = None

    def compute(self):
        sum_x = None
        sum_x_squared = None
        num_samples = 0

        for batch in tqdm(self.dataloader, desc='Computing observed_spectra_std.pt'):
            # Move batch to the appropriate device
            batch = (
                self.augmentations(batch) if self.augmentations is not None else batch
            )
            batch = self.transforms(batch) if self.transforms is not None else batch

            if sum_x is None:
                sum_x = torch.zeros_like(batch[0])
                sum_x_squared = torch.zeros_like(batch[0])

            # Accumulate sum and squared sum
            sum_x += batch.nansum(dim=0)
            sum_x_squared += (batch**2).nansum(dim=0)
            num_samples += batch.size(0)

        # Compute mean and variance
        mean_x = sum_x / num_samples
        mean_x_squared = sum_x_squared / num_samples

        # V = E[X^2] - E[X]^2
        variance = mean_x_squared - mean_x**2

        # Ensure no negative variance due to numerical errors
        variance = torch.clamp(variance, min=0)

        # Compute standard deviation
        self.std_dev = torch.sqrt(variance)

        return self.std_dev


def format_axes(ax: Axes | list[Axes], **kwargs):
    # This handles if two axes are passed in, there is some default styling always done to these
    if isinstance(ax, list) and len(ax) == 2:
        format_axes(
            ax[0], ticks_right=False, **(kwargs[0] if 0 in kwargs.keys() else {})
        )
        format_axes(
            ax[1], ticks_left=False, **(kwargs[1] if 1 in kwargs.keys() else {})
        )

        if 'combine_legends' in kwargs.keys() and kwargs['combine_legends'] is True:
            handles, labels = ax[0].get_legend_handles_labels()
            handles2, labels2 = ax[1].get_legend_handles_labels()

            # Combine the handles and labels
            handles.extend(handles2)
            labels.extend(labels2)

            # into  a single legend
            ax[0].legend(handles, labels)

        return

    if ax.get_legend():
        ax.legend(
            facecolor='white',
            loc='best' if 'legend_loc' not in kwargs.keys() else kwargs['legend_loc'],
        )

    # Make the axes the plots have a white background
    ax.set_facecolor('white')

    # Format the spines
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_edgecolor('k')
        ax.spines[side].set_linewidth(0.5)

    # Add minor ticks to the axes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Turn on all ticks
    ax.tick_params(
        which='both',
        top=True if 'ticks_top' not in kwargs.keys() else kwargs['ticks_top'],
        bottom=True if 'ticks_bottom' not in kwargs.keys() else kwargs['ticks_bottom'],
        left=True if 'ticks_left' not in kwargs.keys() else kwargs['ticks_left'],
        right=True if 'ticks_right' not in kwargs.keys() else kwargs['ticks_right'],
    )

    ax.tick_params(which='minor', length=2, color='k', direction='out')
    ax.tick_params(which='major', length=4, color='k', direction='out')

    if 'autoscale_x' in kwargs.keys() and kwargs['autoscale_x'] is True:
        ax.autoscale(enable=True, tight=True, axis='x')


def save_fig(output_dir: str, name: str, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, name)

    plt.savefig(filename, bbox_inches='tight', **kwargs)

    print('Saved figure to: ', filename)


def load_config(header: str):
    cwd = os.path.dirname(os.path.realpath(__file__))

    with open(f'{cwd}/../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    try:
        return config[header]
    except KeyError:
        raise KeyError(f'Header {header} not found in config file.')
