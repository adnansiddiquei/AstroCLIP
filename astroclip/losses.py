import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, y) -> torch.Tensor:
        """
        Compute the InfoNCE loss between x and y.

        x and y are feature vectors of shape (batch_size, n_features). x[i] and y[i] should correspond to the features
        of the same sample. Via this loss, we want to maximize the similarity between x[i] and y[i] and minimize the
        similarity between x[i] and y[j], where i and j take different values.

        sim = x @ y is the similarity matrix between x and y, where sim[i, j] is the similarity between x[i] and y[j].
        Hence, sim[i, i], the leading diagonal, is the similarity between x[i] and y[i].

        The loss is computed as the average of the cross-entropy loss between the similarity of x and y and the
        similarity of y and x. The InfoNCE loss is simply the cross-entropy loss of the similarity matrix where the
        correct label is the diagonal of the similarity matrix.

        Parameters
        ----------
        x : torch.Tensor
            Feature vectors of shape (batch_size, n_features)
        y : torch.Tensor
            Feature vectors of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            InfoNCE loss.
        """
        assert x.shape == y.shape, 'x and y must have the same shape'

        # Compute cosine similarity
        # The leading diagonal of the similarity matrix is the similarity between x[i] and y[i] as the leading diagonal
        # is the dot product between x[i] and y[i].
        sim = x @ y.T  # shape (batch_size, batch_size)

        # Scale the similarity by the temperature
        logits = sim / self.temperature

        # The labels are the diagonal of the similarity matrix, logits[i, i] is the similarity between x[i] and y[i].
        # So the correct label for x[i] is i. Hence, the below line of code.
        labels = torch.arange(
            logits.shape[0], device=logits.device, dtype=torch.long
        )  # shape (batch_size,)

        # The InfoNCE loss is simply the cross-entropy loss of the similarity matrix where the correct label is the
        # diagonal of the similarity matrix.
        # logits is the similarity matrix between x and y.
        # logits.T is the similarity matrix between y and x, equivalent to logits.T = y @ x.T
        # We compute the cross-entropy loss in both directions and average them.
        # Though in AstroCLIP case, these should really be the same in each direction, as the similarity matrix is
        # symmetric.
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        return loss
