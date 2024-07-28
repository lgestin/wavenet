import torch
import torch.nn as nn
import torch.nn.functional as F

from wavenet.data import MuLaw, Tokenizer
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class WavenetDims:
    n_blocks: int = 3
    n_layers_per_block: int = 10
    in_channels: int = 1
    dim: int = 256
    out_channels: int = 256
    kernel_size: int = 2
    dilation: int = 2


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # s = 1
        # p = [-1 + k + (k-1)*(d-1)]
        pad = -1 + kernel_size + (kernel_size - 1) * (dilation - 1)
        self.pad = pad

    def forward(self, x: torch.Tensor, cache: dict = None):
        if cache is not None:
            cached = cache[self]
            if cached.shape[-1] >= self.cache_size:
                cached = cached[..., 1:]
            x = torch.cat([cached, x], dim=-1)
            cache[self] = x[..., -self.cache_size :]
        else:
            x = F.pad(x, (self.pad, 0))

        x = super().forward(x)
        return x

    @property
    def cache_size(self):
        cache_size = 1 + (self.kernel_size[0] - 1) * self.dilation[0]
        return cache_size


class WavenetLayer(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.dilconv = CausalConv1d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor, cache: dict = None):
        x_res = x
        x = self.dilconv(x, cache=cache)
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

    def forward(self, x: torch.Tensor, cache: dict = None):
        x_skip = 0
        for layer in self.layers:
            x, xs = layer(x, cache=cache)
            x_skip += xs
        return x, x_skip


class Wavenet(nn.Module):
    def __init__(self, dims: WavenetDims):
        super().__init__()
        self.dims = dims

        self.in_conv = nn.Conv1d(
            in_channels=dims.in_channels,
            out_channels=dims.dim,
            kernel_size=1,
            bias=False,
        )

        blocks = []
        for _ in range(dims.n_blocks):
            block = WavenetBlock(
                n_layers=dims.n_layers_per_block,
                dim=dims.dim,
                kernel_size=dims.kernel_size,
                dilation=dims.dilation,
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(dims.dim, dims.dim, 1),
            nn.GELU(),
            nn.Conv1d(dims.dim, dims.out_channels, 1),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, cache: dict = None):
        x = self.in_conv(x)

        x_skip = 0
        for block in self.blocks:
            x, xs = block(x, cache=cache)
            x_skip += xs

        logits = self.out_conv(x_skip)
        return logits.transpose(1, 2)

    @torch.inference_mode()
    def sample(
        self,
        n: int,
        steps: int,
        mulaw: MuLaw,
        tokenizer: Tokenizer,
        verbose: bool = False,
        use_cache: bool = True,
    ):
        cache = None
        if use_cache:
            cache = {}
            for module in self.modules():
                if isinstance(module, CausalConv1d):
                    size = (n, module.in_channels, module.cache_size)
                    cache[module] = torch.zeros(*size, device=self.device)

        sampled = 2 * torch.rand(n, 1, 1, device=self.device) - 1
        pbar = tqdm(desc="SAMPLING", total=steps, leave=False, disable=not verbose)
        for _ in range(steps):
            x_input = sampled if not use_cache else sampled[..., -1:]
            logits = self.forward(x_input, cache=cache)

            # greedy
            p = logits[..., -1, :].softmax(dim=-1)
            s = p.multinomial(1).unsqueeze(1)
            s = mulaw.decode(tokenizer.decode(s))
            sampled = torch.cat([sampled, s], dim=-1)

            pbar.update()

        sampled = sampled[..., 1:]
        return sampled
