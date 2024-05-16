import torch.nn as nn
import torch
import numpy as np
import pytorch_lightning as L


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
