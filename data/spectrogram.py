import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_nfft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window, persistent=False)

    @torch.inference_mode()
    def stft(self, x: torch.FloatTensor):
        pad = (self.n_nfft - self.hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        stft = torch.stft(
            x,
            n_fft=self.n_nfft,
            hop_length=self.hop_length,
            win_length=self.n_nfft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return stft

    def magnitudes(self, x: torch.FloatTensor):
        stft = self.stft(x)
        magnitudes = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2))
        return magnitudes


class MelSpectrogram(STFT):
    def __init__(self, n_mels: int, n_fft: int, hop_length: int, sample_rate: int):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        mel_fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="htk",
        ).transpose(0, 1)
        self.register_buffer("mel_fbank", mel_fbank, persistent=False)

    def mel(self, x: torch.FloatTensor):
        magnitudes = self.magnitudes(x)
        mel = self.mel_fbank @ magnitudes
        return mel
