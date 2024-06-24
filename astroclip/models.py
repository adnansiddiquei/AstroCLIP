"""
This file contains trainable models.

All models in this file are PyTorch Lightning modules.
"""

from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from astroclip.losses import InfoNCELoss
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score


class ContrastiveBimodalPretraining(L.LightningModule):
    def __init__(
        self,
        encoders: list[nn.Module],
        cross_modal_transforms: nn.Sequential = nn.Sequential(),
        train_transforms_and_augmentations: list[nn.Sequential] = (
            nn.Sequential(),
            nn.Sequential(),
        ),
        val_transforms_and_augmentations: list[nn.Sequential] = (
            nn.Sequential(),
            nn.Sequential(),
        ),
        loss: nn.Module = InfoNCELoss(),
        optimizer_kwargs: dict = None,
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
        cross_modal_transforms : nn.Sequential
            ...
        train_transforms_and_augmentations : list[nn.Sequential]
            List of two nn.Sequential objects, one for each modality. These transformations and augmentations will be
            applied to the input data during training. Defaults to two empty nn.Sequential objects.
        val_transforms_and_augmentations : list[nn.Sequential]
            List of two nn.Sequential objects, one for each modality. These transformations and augmentations will be
            applied to the input data during testing. Defaults to two empty nn.Sequential objects.
        loss : nn.Module
            The loss function to use for training. This should be a PyTorch module which takes in two tensors of shape
            (batch_size, embedding_dim) and returns a scalar loss.
        """
        super().__init__()

        assert (
            len(encoders) == 2
        ), 'This module requires exactly two encoders, one for each modality'

        assert (
            len(train_transforms_and_augmentations) == 2
            and len(val_transforms_and_augmentations) == 2
        ), 'transforms_and_augmentations must be a list of length 2, one for each modality'

        self.encoders = nn.ModuleList(encoders)
        self.cross_modal_transforms = [
            cross_modal_transforms
        ]  # list so it isn't saved into the checkpoint
        self.train_transforms_and_augmentations = train_transforms_and_augmentations
        self.val_transforms_and_augmentations = val_transforms_and_augmentations

        self.loss_func = loss

        self.optimizer_kwargs = optimizer_kwargs

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {'lr': 5e-4}

    def _embed(self, modality_idx: int, batch: list[torch.Tensor]):
        # has shape (batch_size, embedding_dim)
        tmp = self.encoders[modality_idx](batch[modality_idx])
        return F.normalize(tmp, p=2, dim=-1)

    def _compute_loss(self, batch: list[torch.Tensor], batch_idx):
        # def _embed(modality_idx: int):
        #     # has shape (batch_size, embedding_dim)
        #     tmp = self.encoders[modality_idx](batch[modality_idx])
        #     return F.normalize(tmp, p=2, dim=-1)

        embedding_0 = self._embed(0, batch)
        embedding_1 = self._embed(1, batch)

        loss = self.loss_func(embedding_0, embedding_1)

        return loss, embedding_0, embedding_1

    def training_step(self, batch: list[torch.Tensor], batch_idx) -> STEP_OUTPUT:
        loss, _, _ = self._compute_loss(batch, batch_idx)
        self.log('train/loss', loss)
        self.log('train/batch_size', batch[0].shape[0])
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx):
        loss, embedding_0, embedding_1 = self._compute_loss(batch, batch_idx)
        self.log('val/loss', loss)
        self.log('val/batch_size', batch[0].shape[0])
        return {
            'val/loss': loss,
            'embedding_0': embedding_0,
            'embedding_1': embedding_1,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        return optimizer

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        def _apply_transforms_and_augmentations(modality_idx: int):
            if self.trainer.training:
                tmp = self.train_transforms_and_augmentations[modality_idx](batch)
            else:
                tmp = self.val_transforms_and_augmentations[modality_idx](batch)

            assert not torch.isnan(
                tmp
            ).any(), f'Modality {modality_idx} contains NaNs after transformations and augmentations have been applied.'

            return tmp

        batch = self.cross_modal_transforms[0](batch)

        return [
            _apply_transforms_and_augmentations(0),
            _apply_transforms_and_augmentations(1),
        ]


class AstroCLIP(ContrastiveBimodalPretraining):
    def __init__(
        self,
        encoders: list[nn.Module],
        val_redshifts: torch.Tensor = None,
        cross_modal_transforms: nn.Sequential = nn.Sequential(),
        train_transforms_and_augmentations: list[nn.Sequential] = (
            nn.Sequential(),
            nn.Sequential(),
        ),
        val_transforms_and_augmentations: list[nn.Sequential] = (
            nn.Sequential(),
            nn.Sequential(),
        ),
        loss: nn.Module = InfoNCELoss(),
        optimizer_kwargs: dict = None,
        checkpoints_dir: str = None,
    ):
        super().__init__(
            encoders=encoders,
            cross_modal_transforms=cross_modal_transforms,
            train_transforms_and_augmentations=train_transforms_and_augmentations,
            val_transforms_and_augmentations=val_transforms_and_augmentations,
            loss=loss,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.checkpoints_dir = checkpoints_dir

        if val_redshifts is not None:
            self.val_redshifts = val_redshifts.numpy()
        else:
            self.val_redshifts = None

    def _predict_redshifts(self, source_embedding, target_embedding, n_neighbours=16):
        """
        Predicts redshift for each embedding in source_embedding, by averaging the redshifts of its closest neighbours
        in target_embedding.
        """
        assert self.val_redshifts is not None, 'Validation redshifts must be provided'

        assert (
            source_embedding.shape[0] == target_embedding.shape[0]
        ), 'Embeddings must have the same dimension'
        assert (
            source_embedding.shape[1] == target_embedding.shape[1]
        ), 'Embeddings must have the same dimension'
        num_embeddings = source_embedding.shape[0]
        actual_redshifts = self.val_redshifts[0 : target_embedding.shape[0]]

        neighbours = NearestNeighbors(n_neighbors=n_neighbours, algorithm='auto').fit(
            target_embedding
        )
        distances, indices = neighbours.kneighbors(source_embedding)
        predicted_redshifts = np.array(
            [actual_redshifts[indices[i]].mean() for i in range(num_embeddings)]
        )
        return actual_redshifts, predicted_redshifts

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # def _apply_transforms_and_augmentations(modality_idx: int):
        #     tmp = self.val_transforms_and_augmentations[modality_idx](batch)
        #
        #     assert not torch.isnan(
        #         tmp
        #     ).any(), f'Modality {modality_idx} contains NaNs after transformations and augmentations have been applied.'
        #
        #     return tmp
        #
        # batch = [
        #     _apply_transforms_and_augmentations(0),
        #     _apply_transforms_and_augmentations(1),
        # ]
        #
        # batch = [
        #     self._embed(0, batch),
        #     self._embed(1, batch)
        # ]
        #
        # return batch

        embedding_0 = self._embed(0, batch)
        embedding_1 = self._embed(1, batch)

        return embedding_0, embedding_1

    def validation_epoch_end(self, outputs) -> None:
        epoch = self.trainer.current_epoch
        image_embeddings = torch.cat([x['embedding_0'] for x in outputs]).to('cpu')
        spectrum_embeddings = torch.cat([x['embedding_1'] for x in outputs]).to('cpu')

        if self.checkpoints_dir:
            torch.save(
                image_embeddings,
                f'{self.checkpoints_dir}/epoch{epoch}_image_embeddings.pt',
            )
            torch.save(
                spectrum_embeddings,
                f'{self.checkpoints_dir}/epoch{epoch}_spectrum_embeddings.pt',
            )

        ii_score = r2_score(
            *self._predict_redshifts(image_embeddings, image_embeddings)
        )
        si_score = r2_score(
            *self._predict_redshifts(spectrum_embeddings, image_embeddings)
        )
        is_score = r2_score(
            *self._predict_redshifts(image_embeddings, spectrum_embeddings)
        )
        ss_score = r2_score(
            *self._predict_redshifts(spectrum_embeddings, spectrum_embeddings)
        )

        self.log('val/ii_score', ii_score)
        self.log('val/si_score', si_score)
        self.log('val/is_score', is_score)
        self.log('val/ss_score', ss_score)
