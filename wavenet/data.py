import torch
import torchaudio
import torch.nn as nn
from dataclasses import dataclass

import torch.utils


@dataclass
class Batch:
    def to(self, device: torch.device):
        for annotation in self.__annotations__:
            value = getattr(self, annotation)
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=True)
                setattr(self, annotation, value)
        return self


@dataclass
class AudioBatch(Batch):
    waveforms: torch.FloatTensor
    tokens: torch.LongTensor

    def __post_init__(self):
        assert self.waveforms.shape[0] == self.tokens.shape[0]
        assert self.waveforms.shape[-1] == self.tokens.shape[-1]


@dataclass
class Audio:
    waveform: torch.FloatTensor
    tokens: torch.LongTensor


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


class LJDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str, mulaw: MuLaw, tokenizer: Tokenizer):
        super().__init__(root, download=True)
        self.mulaw = mulaw
        self.tokenizer = tokenizer

    def __getitem__(self, i: int):
        waveform = super().__getitem__(i)[0]
        tokens = self.tokenizer.encode(self.mulaw.encode(waveform))
        item = Audio(waveform=waveform.unsqueeze(1), tokens=tokens)
        return item


def collate(batch: list[Audio], seq_len: int) -> AudioBatch:
    lengths = [a.waveform.shape[-1] for a in batch]
    assert seq_len < min(lengths)

    idxs = torch.randint(min(lengths) - seq_len, size=(len(batch),))
    waveforms = [a.waveform[..., i : i + seq_len] for i, a in zip(idxs, batch)]
    waveforms = torch.stack(waveforms).squeeze(1)

    tokens = [a.tokens[..., i : i + seq_len] for i, a in zip(idxs, batch)]
    tokens = torch.stack(tokens).squeeze(1)

    return AudioBatch(waveforms=waveforms, tokens=tokens)
