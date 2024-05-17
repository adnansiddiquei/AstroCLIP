import torch


class Roll:
    def __init__(self, mu: float = 0, sigma: float = 1):
        """
        Roll the tensor along the last two dimensions by a random amount equal to a normal distribution with mean `mu`
        and standard deviation `sigma`. The amount of shift is rounded to the nearest integer.

        Parameters
        ----------
        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: torch.Tensor):
        shifts = torch.round((torch.randn(2) * self.sigma) + self.mu).int().tolist()
        return torch.roll(x, shifts, (-1, -2))
