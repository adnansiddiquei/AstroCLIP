import torch.nn as nn
import torch
import numpy as np
import pytorch_lightning as L
import os
import datasets
from datasets import load_dataset


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
