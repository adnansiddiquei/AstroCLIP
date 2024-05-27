"""
This file contains trainable models.

All models in this file are PyTorch Lightning modules.
"""

from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from astroclip.losses import InfoNCELoss


class ContrastiveBimodalPretraining(L.LightningModule):
    def __init__(
        self,
        encoders: list[nn.Module],
        pre_transforms: list[nn.Sequential] = (nn.Sequential(), nn.Sequential()),
        augmentations: list[nn.Sequential] = (nn.Sequential(), nn.Sequential()),
        post_transforms: list[nn.Sequential] = (nn.Sequential(), nn.Sequential()),
        loss: nn.Module = InfoNCELoss(),
        learning_rate=5e-4,
        modality_names: list[str] = ('modality1', 'modality2'),
    ):
        """
        Implementation of a bimodal contrastive pretraining module.

        This module takes two encoders, one for each modality, and trains them to produce similar embeddings for similar
        inputs.

        Parameters
        ----------
        encoders : list[nn.Module]
            List of two encoders, one for each modality. This must be a list of length 2, each element being a PyTorch
            module (the encoders) which takes in an input and produces an embedding. The embeddings produced by the
            encoders must be of the same dimension. If the models are pretrained and only require fine-tuning, the
            relevant weights should be frozen before passing them to this module.
        pre_transforms : list[nn.Sequential | None]
            List of two pre_transforms, one for each modality. This must be a list of length 2, each element being a
            torch.nn.Sequential object which takes in an input and applies a series of transformations to it.
            If no post_transforms are required for a modality, the corresponding element in the list should be an empty
            nn.Sequential. These transformations are applied before the augmentations.
        augmentations : list[nn.Sequential | None]
            List of two augmentations, one for each modality. Formatting is the same as for pre_transforms.
        post_transforms : list[nn.Sequential | None]
            List of two post_transforms, one for each modality. Formatting is the same as for pre_transforms.
        loss : nn.Module
            The loss function to use for training. This should be a PyTorch module which takes in two tensors of shape
            (batch_size, embedding_dim) and returns a scalar loss.
        learning_rate : float
            The learning rate to use for training.
        modality_names : list[str]
            List of two strings, the names of the modalities. These names will be used to access the modalities in the
            batch dictionary. The default is ['modality1', 'modality2']. The order of the names should match the order
            of the encoders, augmentations, and post_transforms. In the case of AstroCLIP, these names would be ['image',
            'spectrum'].
        """
        super().__init__()

        assert (
            len(encoders) == 2
        ), 'This module requires exactly two encoders, one for each modality'

        assert (
            len(pre_transforms) == 2
        ), 'This module requires exactly two pre_transforms, one for each modality'

        assert (
            len(post_transforms) == 2
        ), 'This module requires exactly two post_transforms, one for each modality'

        assert (
            len(augmentations) == 2
        ), 'This module requires exactly two augmentations, one for each modality'

        self.mod1_encoder, self.mod2_encoder = encoders
        self.mod1_pre_transforms, self.mod2_pre_transforms = pre_transforms
        self.mod1_augmentations, self.mod2_augmentations = augmentations
        self.mod1_post_transforms, self.mod2_transforms = post_transforms
        self.mod1_name, self.mod2_name = modality_names

        self.loss_func = loss
        self.learning_rate = learning_rate

    def _compute_loss(self, batch: dict, batch_idx):
        mod1, mod2 = batch[self.mod1_name], batch[self.mod2_name]

        # these both have shape (batch_size, embedding_dim)
        modality1_embeddings = F.normalize(self.mod1_encoder(mod1), p=2, dim=-1)

        modality2_embeddings = F.normalize(self.mod2_encoder(mod2), p=2, dim=-1)

        loss = self.loss_func(modality1_embeddings, modality2_embeddings)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._compute_loss(batch, batch_idx)
        self.log('train/loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log('val/loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # Do some simple checks to ensure that the batch is correctly formatted
        assert isinstance(
            batch, dict
        ), f"Batch must be a dictionary containing the modalities ('{self.mod1_name}', '{self.mod1_name}') as keys"

        assert (
            self.mod1_name in batch.keys()
        ), f"Batch must contain key '{self.mod1_name}'"

        assert (
            self.mod2_name in batch.keys()
        ), f"Batch must contain key '{self.mod1_name}'"

        mod1, mod2 = batch[self.mod1_name], batch[self.mod2_name]

        mod1 = self.mod1_pre_transforms(mod1)
        mod2 = self.mod2_pre_transforms(mod2)

        # Apply augmentations only if training
        if self.trainer.training:
            mod1 = self.mod1_augmentations(mod1)
            mod2 = self.mod2_augmentations(mod2)

        mod1 = self.mod1_post_transforms(mod1)
        mod2 = self.mod2_transforms(mod2)

        assert not torch.isnan(
            mod1
        ).any(), f'{self.mod1_name} contains NaNs after transformations and augmentations have been applied.'

        assert not torch.isnan(
            mod2
        ).any(), f'{self.mod2_name} contains NaNs after transformations and augmentations have been applied.'

        return {self.mod1_name: mod1, self.mod2_name: mod2}


class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        learning_rate=5e-4,
        loss_fn=None,
        pre_transforms=nn.Sequential(),
        post_transforms=nn.Sequential(),
        augmentations=nn.Sequential(),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.augmentations = augmentations

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        clean_batch, augmented_batch = batch['clean_batch'], batch['augmented_batch']
        reconstruction = self(augmented_batch)
        loss = self.loss_fn(reconstruction, clean_batch)
        self.log('train/loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        reconstruction = self(batch)
        loss = self.loss_fn(reconstruction, batch)
        self.log('val/loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch = self.pre_transforms(batch)

        if self.trainer.training:
            # if we are training, then we need to retain a clean non-augmented batch for the loss calculation
            clean_batch = batch.clone()
            clean_batch = self.post_transforms(clean_batch)

            augmented_batch = self.augmentations(batch)
            augmented_batch = self.post_transforms(augmented_batch)

            return {'clean_batch': clean_batch, 'augmented_batch': augmented_batch}
        else:
            batch = self.post_transforms(batch)
            return batch
