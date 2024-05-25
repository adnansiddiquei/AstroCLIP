import os
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

from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.self_supervised import Moco_v2
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from astroclip.models import ContrastiveBimodalPretraining
from astroclip.blocks import SpectrumEncoderSpender
from astroclip.utils import load_config, download_desi_dataset, set_trainable_layers
from astroclip.transforms import Permute, NormaliseSpectrum, Squeeze
from astroclip.augmentations import Roll, AddGaussianNoise, SpectrumNoising


def main():
    # load the config argument to decide which config header to load
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

    args = parser.parse_args()

    # load the config file
    config = load_config(args.config)

    cache_dir = config['cache_dir']
    output_dir = config['output_dir']

    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f'Cache directory {cache_dir} does not exist.')

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'Cache directory {output_dir} does not exist.')

    wandb_logger = WandbLogger(
        log_model='all',
        project='AstroCLIP',
        name=f'train_astroclip_{args.jobid}',
    )

    # Load the dataset, if the dataset is not already in the cache dir it'll be downloaded
    dataset = download_desi_dataset(cache_dir)

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

    # Define all the image and spectrum transforms / augmentations
    image_pre_transforms = nn.Sequential(
        Permute([0, 3, 1, 2]),  # Change to [batch_size, channel, npix, npix]
    )

    image_augmentations = nn.Sequential(
        Roll(0, 3),  # Original paper Rolls by [-5, 5] with uniform distribution
        AddGaussianNoise(0, 0.03),
        RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    )

    image_post_transforms = nn.Sequential(
        CenterCrop(96),
    )

    lambda_min = 3600.0  # Minimum wavelength in Angstroms, in observed spectra
    lambda_max = 9824.0
    z_max = 0.8
    restframe_wavelengths = torch.linspace(lambda_min / (1 + z_max), lambda_max, 7781)

    spectrum_pre_transforms = nn.Sequential(
        Permute([0, 2, 1]),  # Change to [batch_size, channel, spectrum_length]
        NormaliseSpectrum(restframe_wavelengths, (5300, 5850)),
    )

    spectrum_augmentations = nn.Sequential(
        SpectrumNoising(observed_spectra_std_dev),
    )

    spectrum_post_transforms = nn.Sequential(
        Squeeze(
            1
        ),  # Change to [batch_size, spectrum_length], this is how the spender model expects it
    )

    # Load the pre-trained spectrum encoder
    spectrum_encoder = SpectrumEncoderSpender(
        state_dict=torch.load(f'{cache_dir}/spender.desi-edr.encoder.pth'),
        mlp=nn.Sequential(nn.Linear(256, 128)),
    )

    # set only the final MLP layers to be trainable
    set_trainable_layers(spectrum_encoder.encoder, ['mlp'])

    # Load the pre-trained image encoder
    image_model = Moco_v2.load_from_checkpoint(f'{cache_dir}/resnet50.ckpt')
    image_encoder = image_model.encoder_q
    set_trainable_layers(
        image_encoder, ['fc']
    )  # set final fully-connected layer to be trainable

    # define the contrastive learning model
    model = ContrastiveBimodalPretraining(
        encoders=[image_encoder, spectrum_encoder],
        pre_transforms=[image_pre_transforms, spectrum_pre_transforms],
        augmentations=[image_augmentations, spectrum_augmentations],
        post_transforms=[image_post_transforms, spectrum_post_transforms],
        modality_names=['image', 'spectrum'],
    )

    num_workers = max(os.cpu_count(), 1)

    # define dataloaders
    train_loader = DataLoader(
        dataset['train'],
        batch_size=config['astroclip']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset['test'],
        batch_size=config['astroclip']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    # train the model
    model_checkpoints_dir = os.path.join(output_dir, 'astroclip_checkpoints')
    if not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    trainer = L.Trainer(
        max_epochs=config['astroclip']['max_epochs'],
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_checkpoints_dir,
                filename=f'astroclip-{args.jobid}-{{epoch:02d}}-{{val/loss:.2f}}',
                monitor='val/loss',
                mode='min',
            )
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
