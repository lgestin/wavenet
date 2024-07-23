import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # s = 1
        # p = [-1 + k + (k-1)*(d-1)]
        pad = -1 + kernel_size + (kernel_size - 1) * (dilation - 1)
        self.pad = pad

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (self.pad, 0))
        x = super().forward(x)
        return x


class WavenetLayer(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.dilconv = CausalConv(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor):
        x_res = x
        x = self.dilconv(x)
        x_tanh, x_sig = x.chunk(2, dim=1)
        x = x_tanh.tanh() * x_sig.sigmoid()
        x_skip = self.conv(x)
        x = x_res + x_skip
        return x, x_skip


class WavenetBlock(nn.Module):
    def __init__(self, n_layers: int, dim: int, kernel_size: int, dilation: int):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layer = WavenetLayer(
                dim=dim,
                kernel_size=kernel_size,
                dilation=dilation**i,
            )
            layers += [layer]

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        x_skip = 0
        for layer in self.layers:
            x, xs = layer(x)
            x_skip += xs
        return x, x_skip


class Wavenet(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_layers_per_block: int,
        in_channels: int,
        dim: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        self.in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=1,
        )

        blocks = []
        for _ in range(n_blocks):
            block = WavenetBlock(
                n_layers=n_layers_per_block,
                dim=dim,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, out_channels, 1),
        )

    def forward(self, x):
        x = self.in_conv(x)

        x_skip = 0
        for block in self.blocks:
            x, xs = block(x)
            x_skip += xs

        logits = self.out_conv(x)
        return logits
