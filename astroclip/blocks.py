import torch.nn as nn
import spender
from astroclip.utils import copy_weights


class CNNBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.InstanceNorm1d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpectrumEncoderSpender(nn.Module):
    def __init__(self, state_dict=None, mlp=None):
        super(SpectrumEncoderSpender, self).__init__()

        self.encoder = spender.SpectrumEncoder(None, 6)

        if state_dict is not None:
            # load from a state dict if it is provided
            self.encoder.load_state_dict(state_dict, strict=False)

        if mlp is not None:
            # if a different MLP is provided, copy the weights from spender to the new MLP for the first layer
            copy_weights(self.encoder.mlp[0], mlp[0])
            self.encoder.mlp = mlp

    def forward(self, x):
        return self.encoder(x)
