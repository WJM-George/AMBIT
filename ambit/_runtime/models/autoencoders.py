import torch
import math
from torch import nn
from torch.nn.utils import weight_norm
from alias_free_torch import Activation1d
from typing import Literal
from .blocks import SnakeBeta, ResidualUnit


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_activation(
    activation: Literal["elu", "snake", "none"], antialias=False, channels=None
) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    if antialias:
        act = Activation1d(act)
    return act


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False
    ):
        super().__init__()
        self.stride = stride
        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
    ):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            upsample_layer,
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake
            ),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            ]
        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        c_mults = [1] + c_mults
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]
        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioAutoencoder(nn.Module):
    """Continuous FOA VAE with native channels and level preserved."""

    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=4,
        bottleneck=None,
        soft_clip=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.io_channels = io_channels
        self.soft_clip = soft_clip

    def encode(self, audio):
        latent = self.encoder(audio)
        return self.bottleneck.encode(latent) if self.bottleneck is not None else latent

    def decode(self, latent):
        if self.bottleneck is not None:
            latent = self.bottleneck.decode(latent)
        audio = self.decoder(latent)
        return audio.tanh() if self.soft_clip else audio


class VAEBottleneck(nn.Module):
    """Gaussian latent sampling; no training objectives."""

    def encode(self, value):
        (mean, scale) = value.chunk(2, dim=1)
        return mean + torch.randn_like(mean) * (nn.functional.softplus(scale) + 0.0001)

    def decode(self, value):
        return value


class GroupedVAEBottleneck(VAEBottleneck):
    """The grouped codec has the same inference operation and no parameters."""

    def __init__(self, n_w=40):
        super().__init__()
        self.n_w = int(n_w)
