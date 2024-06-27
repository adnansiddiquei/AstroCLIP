import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    CenterCrop,
    InterpolationMode,
)

from pl_bolts.models.self_supervised import Moco_v2

from astroclip.blocks import SpectrumEncoderSpender
from astroclip.utils import (
    download_desi_dataset,
    set_trainable_layers,
)
from astroclip.transforms import (
    Permute,
    Squeeze,
    ExtractKey,
    DropInvalidSpectra,
    DropOnRedshift,
    MeanNormalise,
)
from astroclip.augmentations import Roll, AddGaussianNoise, SpectrumNoising


def parse_args():
    # load the config argument to decide which config config_header to load
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='Which config to use.',
        default='local',
    )

    parser.add_argument(
        '--jobid',
        type=str,
        required=False,
        help='The SLURM job ID, if running on the HPC. Used for logging purposes.',
        default='00000',
    )

    parser.add_argument(
        '--ckptdir',
        type=str,
        required=False,
        help='The directory to store the checkpoints in, relative to the `output_dir` set in the config.yaml. '
        'If this is not provided then no checkpoints will be stored.',
        default=None,
    )

    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Whether to log to WandB or not. Default is True.',
    )

    parser.add_argument(
        '--no-wandb',
        dest='wandb',
        action='store_false',
        help='Disable logging to WandB.',
    )

    parser.add_argument(
        '--hparams',
        type=str,
        required=True,
        help='The hyperparameters to use for training. This should be a string of the relevant config_header in '
        'hyperparams.yaml.',
    )

    parser.set_defaults(wandb=True)

    return parser.parse_args()


def get_image_operations():
    train = nn.Sequential(
        ExtractKey('image'),
        Permute([0, 3, 1, 2]),  # Change to [batch_size, channel, npix, npix]
        Roll(),
        # AddGaussianNoise(0, 0.03),
        RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        CenterCrop(96),
        MeanNormalise(dims=[-1, -2]),  # Normalise the image to mean 0, std 1
        AddGaussianNoise(0, 0.03),
    )

    val = nn.Sequential(
        ExtractKey('image'),
        Permute([0, 3, 1, 2]),  # Change to [batch_size, channel, npix, npix]
        CenterCrop(96),
        MeanNormalise(dims=[-1, -2]),  # Normalise the image to mean 0, std 1
    )

    return train, val


def get_spectrum_operations(output_dir: str):
    # load the observed spectra standard deviation which is generated by compute_observed_spectra_std_dev.py
    try:
        observed_spectra_std_dev = torch.load(
            f'{output_dir}/observed_spectra_std_dev.pt'
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Observed spectra standard deviation file not found in {output_dir}. '
            f'Please run compute_observed_spectra_std_dev.py first.'
        )

    # lambda_min = 3600.0  # Minimum wavelength in Angstroms, in observed spectra
    # lambda_max = 9824.0
    # z_max = 0.8
    # restframe_wavelengths = torch.linspace(lambda_min / (1 + z_max), lambda_max, 7781)

    train = nn.Sequential(
        ExtractKey('spectrum'),
        Permute([0, 2, 1]),  # Change to [batch_size, channel, spectrum_length]
        # NormaliseSpectrum(restframe_wavelengths, (5300, 5850)),
        MeanNormalise(dims=-1),
        SpectrumNoising(observed_spectra_std_dev, 0.3),
        Squeeze(
            1
        ),  # Change to [batch_size, spectrum_length], this is how the spender model expects it
    )

    val = nn.Sequential(
        ExtractKey('spectrum'),
        Permute([0, 2, 1]),  # Change to [batch_size, channel, spectrum_length]
        # NormaliseSpectrum(restframe_wavelengths, (5300, 5850)),
        MeanNormalise(dims=-1),
        Squeeze(
            1
        ),  # Change to [batch_size, spectrum_length], this is how the spender model expects it
    )

    return train, val


def get_cross_modal_transforms():
    batch_transforms = nn.Sequential(
        DropInvalidSpectra(),
        DropOnRedshift(z_min=0.0, z_max=0.8),
    )

    return batch_transforms


def load_pretrained_spectrum_encoder(
    model_path: str, unfreeze_all: bool = False, embedding_dim: int = 128
):
    if embedding_dim <= 128:
        mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
        )
        copy_weights = True
    elif embedding_dim == 256:
        mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        copy_weights = False
    elif embedding_dim == 512:
        mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 512))
        copy_weights = False
    else:
        raise ValueError('Invalid embedding dimension.')

    # Load the pre-trained spectrum encoder
    spectrum_encoder = SpectrumEncoderSpender(
        state_dict=torch.load(model_path), mlp=mlp, copy_mlp_weights=copy_weights
    )

    if not unfreeze_all:
        # set only the final MLP layers to be trainable
        set_trainable_layers(spectrum_encoder.encoder, ['mlp'])

    return spectrum_encoder


def load_pretrained_image_encoder(
    model_path: str, unfreeze_all: bool = False, embedding_dim: int = 128
):
    # Load the pre-trained image encoder
    image_model = Moco_v2.load_from_checkpoint(model_path)
    image_encoder = image_model.encoder_q

    if embedding_dim != 128:
        image_encoder.fc[-1] = nn.Linear(2048, embedding_dim)

    if not unfreeze_all:
        set_trainable_layers(
            image_encoder, ['fc']
        )  # set final fully-connected layer to be trainable

    return image_encoder


def create_dataloaders(cache_dir, batch_size, num_workers, val_drop_last=True):
    # Load the dataset, if the dataset is not already in the cache dir it'll be downloaded
    dataset = download_desi_dataset(cache_dir)

    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        drop_last=val_drop_last,
        num_workers=num_workers,
    )

    return dataset, train_loader, val_loader
