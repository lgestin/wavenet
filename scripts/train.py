import os
import torch

from tqdm import tqdm
from dataclasses import dataclass, asdict
from collections import defaultdict

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader

from wavenet.wavenet import Wavenet, WavenetDims
from data.utils import Tokenizer, MuLaw
from data.dataset import AudioBatch, LJDataset, collate
from data.spectrogram import MelSpectrogram


@dataclass
class TrainingConfig:
    exp_folder: str
    noamp: bool = False
    nocompile: bool = False

    batch_size: int = 16
    lr: float = 3e-4
    q_levels: int = 256

    seq_len: int = 16384
    smp_seq_len: int = 44100
    n_val: int = 128
    n_smp: int = 8
    val_steps: int = 1000
    smp_steps: int = 10000
    max_steps: int = 100000
    n_workers: int = 4


@dataclass
class Checkpoint:
    dims: WavenetDims
    training_config: TrainingConfig

    state_dict: dict[str, torch.Tensor]
    optim_state_dict: dict[str, torch.Tensor]

    step: int
    best_loss: float

    def save(self, save_path: str):
        save_dict = {k: getattr(self, k) for k in self.__annotations__}
        save_dict["dims"] = asdict(save_dict["dims"])
        save_dict["training_config"] = asdict(save_dict["training_config"])
        torch.save(save_dict, save_path)

    @classmethod
    def load(cls, load_path: str):
        checkpoint = torch.load(load_path)
        checkpoint["dims"] = WavenetDims(**checkpoint["dims"])
        checkpoint["training_config"] = TrainingConfig(**checkpoint["training_config"])
        checkpoint = cls(**checkpoint)
        return checkpoint


def train(dims: WavenetDims, training_config: TrainingConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("\nTRAINING CONFIG:")
    print(training_config)
    print("\nMODEL DIMS:")
    print(dims)

    q_levels = training_config.q_levels
    mulaw = MuLaw(q_levels)
    tokenizer = Tokenizer(q_levels)

    dataset = LJDataset("/data/ljspeech", mulaw=mulaw, tokenizer=tokenizer)

    n_smp, n_val = training_config.n_smp, training_config.n_val
    idxs = list(range(len(dataset)))
    smp_dataset = Subset(dataset, idxs[:n_smp])
    val_dataset = Subset(dataset, idxs[n_smp : n_smp + n_val])
    train_dataset = Subset(dataset, idxs[n_smp + n_val :])

    n_workers = min(len(os.sched_getaffinity(0)), training_config.n_workers)
    collate_fn = lambda batch: collate(batch, seq_len=training_config.seq_len + 1)
    smp_dloader = DataLoader(
        smp_dataset,
        batch_size=n_smp,
        num_workers=1,
        collate_fn=collate_fn,
    )
    val_dloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    train_dloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    mel_spectrogram = MelSpectrogram(
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        sample_rate=dataset.sample_rate,
    ).to(device)

    assert dims.out_channels == q_levels
    model = Wavenet(dims=dims)
    model_ = model.to(device)
    model = torch.compile(model_, disable=training_config.nocompile)

    optim = AdamW(model.parameters(), lr=training_config.lr)
    scaler = torch.amp.GradScaler(device)

    def process_batch(batch: AudioBatch):
        metrics = {}
        batch = batch.to(device)

        x_input = batch.waveforms[..., :-1]
        x_target = batch.tokens[..., 1:]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(x_input)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, q_levels), x_target.reshape(-1)
            )

        metrics["loss"] = loss

        if model.training:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e1)
            scaler.step(optim)
            scaler.update()
            metrics["grad"] = grad

        metrics = {k: v.item() for k, v in metrics.items()}
        return metrics

    writer = SummaryWriter(training_config.exp_folder)

    def log_waveform(waveform: torch.FloatTensor, tag: str, step: int):
        waveform = waveform.to(device)

        mels = mel_spectrogram.mel(waveform).log()
        mels = (mels - mels.min()) / (mels.max() - mels.min())
        writer.add_audio(
            f"{tag}.wav", waveform, global_step=step, sample_rate=dataset.sample_rate
        )
        writer.add_image(f"{tag}.mel", mels.flip(1), step)

    # Log original files
    for sbatch in smp_dloader:
        for i, waveform in enumerate(sbatch.waveforms):
            # original
            waveform = waveform.to(device)
            log_waveform(waveform, tag=f"orig/{i}", step=0)

            # encoded waveform
            waveform = mulaw.decode(
                tokenizer.decode(tokenizer.encode(mulaw.encode(waveform)))
            )
            log_waveform(waveform, tag=f"encoded/{i}", step=0)
            writer.add_histogram("orig", sbatch.tokens.reshape(-1), 0)

    pbar = tqdm(total=training_config.max_steps, unit=f"batch")
    step, best_loss = 0, torch.inf
    while 1:
        for batch in train_dloader:

            # Sampling
            if step % training_config.smp_steps == 0:
                model = model.eval()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        sampled = model_.sample(
                            n=n_smp,
                            steps=training_config.smp_seq_len,
                            mulaw=mulaw,
                            tokenizer=tokenizer,
                            verbose=True,
                            use_cache=True,
                        )
                for i, smp in enumerate(sampled):
                    log_waveform(smp, tag=f"smp/{i}", step=step)

            # Validation
            if step % training_config.val_steps == 0:
                model = model.eval()
                vmetrics = defaultdict(list)
                for vbatch in tqdm(val_dloader, desc="VALIDATION", leave=False):
                    with torch.inference_mode():
                        batch_vmetrics = process_batch(vbatch)

                    for k, v in batch_vmetrics.items():
                        vmetrics[k] += [torch.as_tensor(v)]
                vmetrics = {
                    k: torch.stack(v).mean().item() for k, v in vmetrics.items()
                }
                for k, v in vmetrics.items():
                    writer.add_scalar(f"val/{k}", v, step)

                # Teacher forced samples for monitoring
                with torch.inference_mode():
                    waveforms = sbatch.waveforms[..., :-1].to(device)
                    sampled = model(waveforms).softmax(-1).argmax(-1)
                writer.add_histogram(f"val", sampled.reshape(-1), step)
                sampled = mulaw.decode(tokenizer.decode(sampled))
                for i, s in enumerate(sampled):
                    log_waveform(s.unsqueeze(0), tag=f"val/{i}", step=step)

                # Save Checkpoints
                checkpoint = Checkpoint(
                    dims=dims,
                    training_config=training_config,
                    state_dict=model_.state_dict(),
                    optim_state_dict=optim.state_dict(),
                    step=step,
                    best_loss=best_loss,
                )
                checkpoint.save(
                    os.path.join(training_config.exp_folder, "checkpoint.last.pt")
                )
                if vmetrics["loss"] < best_loss:
                    best_loss = vmetrics["loss"]
                    checkpoint.save(
                        os.path.join(training_config.exp_folder, "checkpoint.best.pt")
                    )

            # Training
            model = model.train()
            metrics = process_batch(batch)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, step)

            step += 1
            pbar.set_description_str(f"TRAINING | Loss {metrics['loss']:.4f}")
            pbar.update(1)

            if step > training_config.max_steps:
                return


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(WavenetDims, dest="dims")
    parser.add_arguments(TrainingConfig, dest="training_config")

    args = parser.parse_args()

    train(dims=args.dims, training_config=args.training_config)
