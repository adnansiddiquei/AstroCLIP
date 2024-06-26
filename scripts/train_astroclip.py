import os

import numpy as np
import torch

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from astroclip.models import AstroCLIP
from astroclip.utils import (
    load_config,
    remove_empty_keys,
)

from astroclip.training_utils import (
    parse_args,
    get_image_operations,
    get_spectrum_operations,
    load_pretrained_spectrum_encoder,
    load_pretrained_image_encoder,
    create_dataloaders,
    get_cross_modal_transforms,
)


def main():
    # set the seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()

    if args.config == 'hpc':
        torch.set_float32_matmul_precision('high')

    # load the config file
    config, hparams = load_config(args.config, args.hparams)

    cache_dir = config['cache_dir']
    output_dir = config['output_dir']

    # TODO: batch_transforms should really be applied to the entire dataset, and not a batch at a time. This will result
    # in inconsistent batch sizes (max 1% fluctuation in batch size)
    cross_modal_transforms = get_cross_modal_transforms()

    img_train_transforms, img_val_transforms = get_image_operations()

    spectrum_train_transforms, spectrum_val_transforms = get_spectrum_operations(
        output_dir
    )

    spectrum_encoder = load_pretrained_spectrum_encoder(
        f'{cache_dir}/{config["pretrained_spectrum_encoder"]}',
        unfreeze_all=hparams['unfreeze_all'],
        embedding_dim=hparams['embedding_dim'],
    )

    image_encoder = load_pretrained_image_encoder(
        f'{cache_dir}/{config["pretrained_image_encoder"]}',
        unfreeze_all=hparams['unfreeze_all'],
        embedding_dim=hparams['embedding_dim'],
    )

    model_checkpoints_dir = f'{output_dir}/{args.ckptdir}' if args.ckptdir else None

    if args.ckptdir and not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    dataset, train_loader, val_loader = create_dataloaders(
        cache_dir, hparams['batch_size'], config['num_workers']
    )

    # Get the valid redshifts for the validation set, this is so we can compute and track the R-squared values of the
    # redshift predictions as we go
    if os.path.exists(f'{cache_dir}/val_redshifts.pt'):
        val_redshifts = torch.load(f'{cache_dir}/val_redshifts.pt')
    else:
        val_redshifts = torch.Tensor([])
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = cross_modal_transforms(batch)
                val_redshifts = torch.cat([val_redshifts, batch['redshift']])
        torch.save(val_redshifts, f'{cache_dir}/val_redshifts.pt')

    # define the contrastive learning model
    model = AstroCLIP(
        encoders=[image_encoder, spectrum_encoder],
        cross_modal_transforms=cross_modal_transforms,
        train_transforms_and_augmentations=[
            img_train_transforms,
            spectrum_train_transforms,
        ],
        val_transforms_and_augmentations=[img_val_transforms, spectrum_val_transforms],
        optimizer_kwargs={'lr': hparams['lr'], 'weight_decay': hparams['weight_decay']},
        checkpoints_dir=model_checkpoints_dir,
        val_redshifts=val_redshifts,
    )

    # These are passed into the L.Trainer class, but whether they are passed in are optional based on the flags
    # that were passed in when this python script was called
    trainer_kwargs = remove_empty_keys(
        {
            # log to WandB if the flag is set, otherwise, don't log to WanDB
            'logger': WandbLogger(
                log_model='all',
                project='AstroCLIP',
                name=f'train_astroclip_{args.jobid}',
            )
            if args.wandb
            else None,
            # save model checkpoints if the directory is provided, otherwise don't save checkpoints
            'callbacks': [
                ModelCheckpoint(
                    dirpath=model_checkpoints_dir,
                    filename='astroclip-{epoch:02d}-min',
                    monitor='val/loss',
                    mode='min',
                ),
                ModelCheckpoint(
                    dirpath=model_checkpoints_dir,
                    filename='astroclip-{epoch}-last',
                    save_last=True,
                ),
            ]
            if args.ckptdir
            else None,
        }
    )

    if args.wandb:
        trainer_kwargs['logger'].log_hyperparams(hparams)

    trainer = L.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator='auto',
        devices='auto',
        **trainer_kwargs,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
