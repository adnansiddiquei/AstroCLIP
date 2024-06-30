import numpy as np
import torch

import pytorch_lightning as L
from tqdm import tqdm

from astroclip.models import AstroCLIP
from astroclip.utils import load_config, create_dir_if_required

from astroclip.training_utils import (
    get_image_operations,
    get_spectrum_operations,
    load_pretrained_spectrum_encoder,
    load_pretrained_image_encoder,
    create_dataloaders,
    get_cross_modal_transforms,
)

import argparse


def main():
    embedding_output_dir = create_dir_if_required(__file__, 'out')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='Which config to use.',
        default='local',
    )

    parser.add_argument(
        '--model-checkpoint',
        type=str,
        required=True,
        help='The name of the model checkpoint to load. The checkpoint must be stored in the cache directory.',
    )

    args = parser.parse_args()

    config, hparams = load_config(args.config, 'generate_embeddings')

    cache_dir = config['cache_dir']
    output_dir = config['output_dir']
    embedding_dim = hparams['embedding_dim']

    # Get the encoders
    spectrum_encoder = load_pretrained_spectrum_encoder(
        f'{cache_dir}/{config["pretrained_spectrum_encoder"]}',
        embedding_dim=embedding_dim,
    )

    image_encoder = load_pretrained_image_encoder(
        f'{cache_dir}/{config["pretrained_image_encoder"]}', embedding_dim=embedding_dim
    )

    cross_modal_transforms = get_cross_modal_transforms()

    img_train_transforms, img_val_transforms = get_image_operations()

    spectrum_train_transforms, spectrum_val_transforms = get_spectrum_operations(
        output_dir
    )

    dataset, train_loader, val_loader = create_dataloaders(
        cache_dir,
        hparams['batch_size'],
        config['num_workers'],
        val_drop_last=False,
    )

    model = AstroCLIP.load_from_checkpoint(
        f'{cache_dir}/{args.model_checkpoint}',
        encoders=[image_encoder, spectrum_encoder],
        train_transforms_and_augmentations=[
            img_train_transforms,
            spectrum_train_transforms,
        ],
        val_transforms_and_augmentations=[img_val_transforms, spectrum_val_transforms],
    )

    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
    )

    embeddings = trainer.predict(model, dataloaders=val_loader)
    image_embeddings = torch.concat([e[0] for e in embeddings], dim=0).cpu()
    spectrum_embeddings = torch.concat([e[1] for e in embeddings], dim=0).cpu()

    torch.save(image_embeddings, f'{embedding_output_dir}/image_embeddings.pt')
    torch.save(spectrum_embeddings, f'{embedding_output_dir}/spectrum_embeddings.pt')

    valid_idx = torch.Tensor([]).to(torch.int64)
    batch_size = hparams['batch_size']

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            valid_idx_in_current_batch = torch.from_numpy(
                np.nonzero(
                    np.in1d(
                        batch['targetid'], cross_modal_transforms(batch)['targetid']
                    )
                )[0]
                + idx * batch_size
            ).to(torch.int64)

            valid_idx = torch.cat([valid_idx, valid_idx_in_current_batch])

    torch.save(valid_idx, f'{embedding_output_dir}/valid_indices.pt')


if __name__ == '__main__':
    main()
