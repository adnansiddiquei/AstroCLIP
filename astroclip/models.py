"""
This file contains the AstroCLIP implementation and the SpectrumEncoderSpender model which is simply a wrapper
around the spender.SpectrumEncoder model by Liang, Melchior, et al. (2023
"""

from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from astroclip.losses import InfoNCELoss
from astroclip.utils import copy_weights
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
import spender


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

        This class can work standalone to train any two arbitrary encoders to produce aligned embeddings. However,
        this class is used as a base class for the AstroCLIP class, which is a specific implementation of this class
        which adds some additional AstroCLIP related logging for the report.

        This class expects the batches to be input as a dict with least two keys, each key referring to the
        batch of data for each modality.

        Parameters
        ----------
        encoders : list[nn.Module]
            List of two encoders, one for each modality. This must be a list of length 2, each element being a PyTorch
            module (the encoders) which takes in an input and produces an embedding. The embeddings produced by the
            encoders must be of the same dimension. If the models are pretrained and only require fine-tuning, the
            relevant weights should be frozen before passing them to this module.
        cross_modal_transforms : nn.Sequential
            These transforms are applied to the input data before train_transforms_and_augmentations or
            val_transforms_and_augmentations are applied, and will have access to the entire batch and so can apply
            transforms that require information from both modalities.
        train_transforms_and_augmentations : list[nn.Sequential]
            List of two nn.Sequential objects, one for each modality. These transformations and augmentations will be
            applied to the input data during training. This should make use to astroclip.transforms.ExtractKey
            to extract the relevant modality from the input data.
        val_transforms_and_augmentations : list[nn.Sequential]
            List of two nn.Sequential objects, one for each modality. These transformations and augmentations will be
            applied to the input data during testing. This should make use to astroclip.transforms.ExtractKey
            to extract the relevant modality from the input data.
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
        """
        Embed the given batch using the encoder for the specified modality.

        Parameters
        ----------
        modality_idx : int
            A list of two tensors, each tensor containing the batch for each modality.
        batch : list[torch.Tensor]
            The entire batch of data for both modalities.

        Returns
        -------
        torch.Tensor
            The embedding for the specified modality.
        """
        # has shape (batch_size, embedding_dim)
        tmp = self.encoders[modality_idx](batch[modality_idx])
        return F.normalize(tmp, p=2, dim=-1)

    def _compute_loss(
        self, batch: list[torch.Tensor], batch_idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the contrastive loss for the given batch.

        Parameters
        ----------
        batch : list[torch.Tensor]
            A list of two tensors, each tensor containing the batch for each modality.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The loss, the embedding for the first modality, and the embedding for the second modality.
        """
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
    def on_after_batch_transfer(
        self, batch: Any, dataloader_idx: int
    ) -> list[torch.Tensor]:
        """
        This method is called after the batch is transferred to the correct device. This method applies all the
        transformations and augmentations to the batch.

        Parameters
        ----------
        batch : dict
            The batch of data to be processed. This should be a dict containing the data for both modalities.
        dataloader_idx : int
            The index of the dataloader that the batch is from.

        Returns
        -------
        list[torch.Tensor]
            A list of two tensors, one for each modality, which have had the transformations and augmentations applied.
            The first entry contains the transformed and augmented batch for the first modality, and the second entry
            contains the transformed and augmented batch for the second modality.
        """

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
        """
        Implementation of the AstroCLIP model.

        This class is a specific implementation of the ContrastiveBimodalPretraining class which adds some additional
        AstroCLIP related logging for the report.
        """
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

    def _predict_redshifts(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        n_neighbours=16,
    ):
        """
        Predicts redshift for each embedding in source_embedding, by averaging the redshifts of its closest neighbours
        in target_embedding.

        Parameters
        ----------
        source_embedding : torch.Tensor
            The embeddings for which to predict redshifts.
        target_embedding : torch.Tensor
            The embeddings to use to predict the redshifts.
        n_neighbours : int
            The number of neighbours to use for the prediction.
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
        embedding_0 = self._embed(0, batch)
        embedding_1 = self._embed(1, batch)

        return embedding_0, embedding_1

    def validation_epoch_end(self, outputs) -> None:
        """
        This method is called at the end of the validation epoch. It computes the R^2 scores for all the different
        combinations of embeddings and logs them.

        Parameters
        ----------
        outputs : list
            A list of dictionaries containing the outputs from each validation step.
        """
        epoch = self.trainer.current_epoch
        image_embeddings = torch.cat([x['embedding_0'] for x in outputs]).to('cpu')
        spectrum_embeddings = torch.cat([x['embedding_1'] for x in outputs]).to('cpu')

        # save the embeddings if a checkpoints directory is provided
        if self.checkpoints_dir:
            torch.save(
                image_embeddings,
                f'{self.checkpoints_dir}/epoch{epoch}_image_embeddings.pt',
            )
            torch.save(
                spectrum_embeddings,
                f'{self.checkpoints_dir}/epoch{epoch}_spectrum_embeddings.pt',
            )

        # Compute the R^2 scores for all the different combinations of cross-modal and in-modal redshift predictions
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

        # Log the R^2 scores
        self.log('val/ii_score', ii_score)
        self.log('val/si_score', si_score)
        self.log('val/is_score', is_score)
        self.log('val/ss_score', ss_score)


class SpectrumEncoderSpender(nn.Module):
    def __init__(self, state_dict=None, mlp=None, copy_mlp_weights=True):
        """
        Wrapper around spender.SpectrumEncoder model.

        The purpose of this class is simply to load the spender.SpectrumEncoder model and optionally replace the MLP
        with a different MLP. This is useful for loading a pre-trained model and replacing the MLP with a different one
        to change the dimensionality of the output embedding.

        Parameters
        ----------
        state_dict : dict, optional
            The state dict for the model. If this is provided, the model will be loaded from this state dict.
        mlp : nn.Sequential, optional
            The MLP to use for the model. If this is provided, the MLP in the model will be replaced with this MLP.
        copy_mlp_weights : bool
            If True, the weights from the first layer of the provided MLP will be copied to the first layer of the
            MLP in the model.
        """
        super(SpectrumEncoderSpender, self).__init__()

        self.encoder = spender.SpectrumEncoder(None, 6)

        if state_dict is not None:
            # load from a state dict if it is provided
            self.encoder.load_state_dict(state_dict, strict=False)

        if mlp is not None:
            self.encoder.mlp = mlp

        if copy_mlp_weights:
            # if a different MLP is provided, copy the weights from spender to the new MLP for the first layer
            copy_weights(self.encoder.mlp[0], mlp[0])

    def forward(self, x):
        return self.encoder(x)
