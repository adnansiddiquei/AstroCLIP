import torch
import torch.nn as nn
import spender


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


class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        n_latent=128,
        input_channels=1,
        n_channels=(128, 256, 512),
        kernel_sizes=(5, 11, 21),
        strides=(1, 1, 1),
        n_hidden=(512, 256, 128),
        act=None,
        dropout=0,
    ):
        super(SpectrumEncoder, self).__init__()
        self.n_latent = n_latent
        self.n_channels = n_channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.n_hidden = n_hidden
        self.input_channels = input_channels

        self.conv_blocks = nn.ModuleList()
        last = self.input_channels

        for channels, kernel_size, stride in zip(
            self.n_channels, self.kernel_sizes, self.strides
        ):
            self.conv_blocks.append(
                CNNBlock1d(
                    last, channels, kernel_size, stride, padding=kernel_size // 2
                )
            )

            self.conv_blocks.append(
                nn.MaxPool1d(kernel_size, stride=stride, padding=kernel_size // 2)
            )

            last = channels

        # remove the last pooling layer
        del self.conv_blocks[-1]

        self.softmax = nn.Softmax(dim=-1)

        # create the MLP which will take the attended features and produce the final latents
        if act is None:
            act = [nn.PReLU(n) for n in self.n_hidden]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())

        self.mlp = spender.MLP(
            self.n_channels[-1] // 2,
            self.n_latent,
            n_hidden=self.n_hidden,
            act=act,
            dropout=dropout,
        )

    def forward(self, x):
        # Start with the convolutional blocks, x starts at shape (N=1024, C=1, L=7781)
        for block in self.conv_blocks:
            x = block(x)
        # x exits the block as a tensor of shape (N=1024, C=512, L=7781)

        C = x.shape[1] // 2  # split half channels into attention value and key
        h, a = torch.split(
            x, [C, C], dim=1
        )  # h, a are tensors of shape (N=1024, C=256, L=7781)

        # softmax attention
        a = self.softmax(a)

        # apply attention, produces tensor of shape (N=1024, C=256)
        x = torch.sum(h * a, dim=2)

        # run attended features into MLP for final latents
        x = self.mlp(x)

        return x


class SpectrumDecoder(nn.Module):
    def __init__(
        self,
        n_latent=128,
        n_hidden=(256, 512, 1024),
        spectral_dim=7781,
        act=None,
        dropout=0.0,
    ):
        super(SpectrumDecoder, self).__init__()

        if act is None:
            act = [spender.SpeculatorActivation(n) for n in n_hidden]
            act.append(spender.SpeculatorActivation(spectral_dim, plus_one=True))

        self.n_latent = n_latent

        self.mlp = spender.MLP(
            n_latent,
            spectral_dim,
            n_hidden=n_hidden,
            act=act,
            dropout=dropout,
        )

        self.n_latent = n_latent

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(x.shape[0], 1, -1)
        return x
