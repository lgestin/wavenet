import os
import torch

from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader

from wavenet.wavenet import Wavenet, WavenetDims
from wavenet.data import LJDataset, Tokenizer, MuLaw, collate


@dataclass
class TrainingConfig:
    exp_folder: str
    noamp: bool = False
    nocompile: bool = False

    batch_size: int = 16
    lr: float = 3e-4
    q_levels: int = 256

    seq_len: int = 8192
    n_val: int = 128
    n_smp: int = 8
    val_steps: int = 1000
    smp_steps: int = 10000
    max_steps: int = 100000
    n_workers: int = 4


@dataclass
class Checkpoint:
    state_dict: dict[str, torch.Tensor]
    optim_state_dict: dict[str, torch.Tensor]

    step: int
    best_loss: float

    def save(self, save_path: str):
        torch.save({k: getattr(self, k) for k in self.__annotations__}, save_path)

    @classmethod
    def load(cls, load_path: str):
        checkpoint = torch.load(load_path)
        checkpoint = cls(**checkpoint)
        return checkpoint


def train(dims: WavenetDims, training_config: TrainingConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    assert dims.out_channels == q_levels
    model = Wavenet(dims=dims)
    model = model.to(device)
    model = torch.compile(model, disable=training_config.nocompile)

    optim = AdamW(model.parameters(), lr=training_config.lr)
    scaler = torch.cuda.amp.GradScaler()

    def process_batch(batch: torch.Tensor):
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

    for sbatch in smp_dloader:
        for i, waveform in enumerate(sbatch.waveforms):
            writer.add_audio(f"orig/{i}.wav", waveform, 0, sample_rate=22050)
            waveform = mulaw.decode(
                tokenizer.decode(tokenizer.encode(mulaw.encode(waveform)))
            )
            writer.add_audio(f"encoded/{i}.wav", waveform, 0, sample_rate=22050)
            writer.add_histogram("orig", sbatch.tokens.reshape(-1), 0)

    pbar = tqdm(desc="TRAINING", total=training_config.max_steps)
    step, best_loss = 0, torch.inf
    while step < training_config.max_steps:
        for batch in train_dloader:

            # Sampling
            if step % training_config.smp_steps == training_config.smp_steps - 1:
                model = model.eval()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        sampled = model.sample(
                            n=n_smp,
                            steps=training_config.seq_len,
                            mulaw=mulaw,
                            tokenizer=tokenizer,
                            verbose=True,
                        )
                for i, smp in enumerate(sampled):
                    smp = 0.8 * (2 * (smp - smp.min()) / (smp.max() - smp.min()) - 1)
                    writer.add_audio(f"smp/{i}.wav", smp, step, sample_rate=22050)

            # Validation
            if step % training_config.val_steps == 0:
                model = model.eval()
                vmetrics = defaultdict(list)
                for vbatch in tqdm(val_dloader, desc="VALIDATION", leave=False):
                    batch_vmetrics = process_batch(vbatch)
                    for k, v in batch_vmetrics.items():
                        vmetrics[k] += [torch.as_tensor(v)]
                vmetrics = {
                    k: torch.stack(v).mean().item() for k, v in vmetrics.items()
                }

                for k, v in vmetrics.items():
                    writer.add_scalar(f"val/{k}", v, step)

                with torch.inference_mode():
                    waveforms = sbatch.waveforms[..., :-1].to(device)
                    sampled = model(waveforms).softmax(-1).argmax(-1)
                writer.add_histogram(f"val", sampled.reshape(-1), step)

                checkpoint = Checkpoint(
                    state_dict=model.state_dict(),
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
            pbar.update()
    return


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(WavenetDims, dest="dims")
    parser.add_arguments(TrainingConfig, dest="training_config")

    args = parser.parse_args()

    train(dims=args.dims, training_config=args.training_config)
