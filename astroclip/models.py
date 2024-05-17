from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms import Compose


from .losses import InfoNCELoss


class ContrastiveBimodalPretraining(L.LightningModule):
    def __init__(
        self,
        encoders: list[nn.Module],
        augmentations: list[Compose | None] = (None, None),
        transforms: list[Compose | None] = (None, None),
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
        augmentations : list[Compose | None]
            List of two augmentations, one for each modality. This must be a list of length 2, each element being a
            torchvision.transforms.Compose object which takes in an input and applies a series of transformations to it.
            If no augmentations are required for a modality, the corresponding element in the list should be None.
        transforms : list[Compose | None]
            List of two transforms, one for each modality. This must be a list of length 2, each element being a
            torchvision.transforms.Compose object which takes in an input and applies a series of transformations to it.
            If no transforms are required for a modality, the corresponding element in the list should be None.
        loss : nn.Module
            The loss function to use for training. This should be a PyTorch module which takes in two tensors of shape
            (batch_size, embedding_dim) and returns a scalar loss.
        learning_rate : float
            The learning rate to use for training.
        modality_names : list[str]
            List of two strings, the names of the modalities. These names will be used to access the modalities in the
            batch dictionary. The default is ['modality1', 'modality2']. The order of the names should match the order
            of the encoders, augmentations, and transforms. In the case of AstroCLIP, these names would be ['image',
            'spectrum'].
        """
        super().__init__()

        assert (
            len(encoders) == 2
        ), 'This module requires exactly two encoders, one for each modality'

        assert (
            len(transforms) == 2
        ), 'This module requires exactly two transforms, one for each modality'

        assert (
            len(augmentations) == 2
        ), 'This module requires exactly two augmentations, one for each modality'

        self.mod1_encoder, self.mod2_encoder = encoders
        self.mod1_augmentations, self.mod2_augmentations = augmentations
        self.mod1_transforms, self.mod2_transforms = transforms
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
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _apply_augmentations(self, batch) -> dict:
        mod1, mod2 = batch[self.mod1_name], batch[self.mod2_name]

        mod1 = (
            self.mod1_augmentations(mod1)
            if self.mod1_augmentations is not None
            else mod1
        )

        mod2 = (
            self.mod2_augmentations(mod2)
            if self.mod2_augmentations is not None
            else mod2
        )

        return {self.mod1_name: mod1, self.mod2_name: mod2}

    def _apply_transforms(self, batch) -> dict:
        mod1, mod2 = batch[self.mod1_name], batch[self.mod2_name]

        mod1 = self.mod1_transforms(mod1) if self.mod1_transforms is not None else mod1

        mod2 = self.mod2_transforms(mod2) if self.mod2_transforms is not None else mod2

        return {self.mod1_name: mod1, self.mod2_name: mod2}

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

        # Apply augmentations and transforms to the modalities
        if self.trainer.training:
            # perform the augmentations only if training, we do not want to augment the data during validation
            batch = self._apply_augmentations(batch)

        batch = self._apply_transforms(batch)

        return batch
