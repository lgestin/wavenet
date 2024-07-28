import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, mu: int):
        super().__init__()
        boundaries = torch.linspace(-1, 1, mu)
        self.register_buffer("boundaries", boundaries, persistent=False)

    @torch.inference_mode
    def encode(self, x: torch.FloatTensor) -> torch.LongTensor:
        boundaries = self.boundaries.to(x.device)
        x = torch.bucketize(x, boundaries)
        return x

    @torch.inference_mode
    def decode(self, x: torch.LongTensor) -> torch.FloatTensor:
        boundaries = self.boundaries.to(x.device)
        x = boundaries[x]
        return x

    @property
    def atol(self):
        return self.boundaries[1] - self.boundaries[0]


class MuLaw(nn.Module):
    def __init__(self, mu: int = 256):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu - 1), persistent=False)

    def forward(self, x: torch.Tensor):
        return self.encode(x)

    @torch.inference_mode
    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert torch.all(x.abs().max() <= 1)
        mu = self.mu.to(x.device)
        x = x.sign() * torch.log(1 + mu * x.abs()) / torch.log(1 + mu)
        return x

    @torch.inference_mode
    def decode(self, y: torch.FloatTensor) -> torch.FloatTensor:
        assert torch.all(y.abs().max() <= 1)
        mu = self.mu.to(y.device)
        y = y.sign() * ((1 + mu).pow(y.abs()) - 1) / mu
        return y
