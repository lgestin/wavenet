import torch
import torchaudio
from dataclasses import dataclass

from data.utils import Tokenizer, MuLaw


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


class LJDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root: str, mulaw: MuLaw, tokenizer: Tokenizer):
        super().__init__(root, download=True)
        self.mulaw = mulaw
        self.tokenizer = tokenizer
        self.sample_rate = 22050

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
