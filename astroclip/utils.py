import torch.nn as nn
import torch
import numpy as np
import pytorch_lightning as L
import os
import datasets
from datasets import load_dataset


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
    cwd = os.path.dirname(os.path.realpath(__file__))

    dataset_dir = f'{cache_dir}/datasets_astroclip'

    dataset = load_dataset(
        f'{cwd}/legacy_survey.py',
        cache_dir=dataset_dir,
        trust_remote_code=True,
    )
    dataset.set_format(type='torch')

    return dataset
