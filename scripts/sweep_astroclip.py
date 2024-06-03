import os

import torch
import torch.nn as nn
import wandb

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as L

from astroclip.models import ContrastiveBimodalPretraining
from astroclip.utils import (
    load_config,
)

from astroclip.training_utils import (
    parse_args,
    get_image_operations,
    get_spectrum_operations,
    load_pretrained_spectrum_encoder,
    load_pretrained_image_encoder,
    create_dataloaders,
    get_batch_transforms,
)

from astroclip.losses import InfoNCELoss
from astroclip.augmentations import SpectrumNoising


def train_model():
    wandb.init(project='AstroCLIP')
    wandb_config = wandb.config

    args = parse_args()

    if args.config == 'hpc':
        torch.set_float32_matmul_precision('high')

    print('jobid:', args.jobid)

    # load the config file
    config = load_config(args.config)

    cache_dir = config['cache_dir']
    output_dir = config['output_dir']

    batch_transforms = get_batch_transforms()

    (
        image_pre_transforms,
        image_augmentations,
        image_post_transforms,
    ) = get_image_operations()

    (
        spectrum_pre_transforms,
        spectrum_augmentations,
        spectrum_post_transforms,
    ) = get_spectrum_operations(output_dir)

    spectrum_augmentations = nn.Sequential(
        SpectrumNoising(
            spectrum_augmentations[0].observed_spectra_std_dev,
            noise_strength=wandb_config.spectrum_noise_strength,
        ),
    )

    spectrum_encoder = load_pretrained_spectrum_encoder(
        config['astroclip']['pretrained_spectrum_encoder']
    )

    image_encoder = load_pretrained_image_encoder(
        config['astroclip']['pretrained_image_encoder']
    )

    # define the contrastive learning model
    model = ContrastiveBimodalPretraining(
        encoders=[image_encoder, spectrum_encoder],
        batch_transforms=batch_transforms,
        pre_transforms=[image_pre_transforms, spectrum_pre_transforms],
        augmentations=[image_augmentations, spectrum_augmentations],
        post_transforms=[image_post_transforms, spectrum_post_transforms],
        optimizer_kwargs={'lr': 5e-4, 'weight_decay': wandb_config.weight_decay},
        loss=InfoNCELoss(temperature=wandb_config.temperature),
    )

    train_loader, val_loader = create_dataloaders(
        cache_dir, config['astroclip']['batch_size'], config['astroclip']['num_workers']
    )

    # train the model
    model_checkpoints_dir = f'{output_dir}/{args.ckptdir}'

    if not os.path.exists(model_checkpoints_dir) and args.ckptdir:
        os.makedirs(model_checkpoints_dir)

    trainer = L.Trainer(
        max_epochs=30,
        accelerator='auto',
        devices='auto',
        logger=WandbLogger(
            log_model='all',
            project='AstroCLIP',
            name=f'train_astroclip_{args.jobid}',
        ),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val/loss', 'goal': 'minimize'},
        'parameters': {
            'weight_decay': {'values': [1e-5, 1e-3, 1e-1, 1.0, 10.0]},
            'temperature': {'values': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            'spectrum_noise_strength': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='AstroCLIP')
    print(f'sweep_id: {sweep_id}')

    wandb.agent(sweep_id=sweep_id, function=train_model, count=10)
