import os
import torch
from wavenet.wavenet import Wavenet
from scripts.train import Checkpoint, TrainingConfig
from data.utils import Tokenizer, MuLaw


def sample(checkpoint_path: str, save_folder: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = Checkpoint.load(checkpoint_path)

    q_levels = checkpoint.training_config.q_levels
    mulaw = MuLaw(q_levels)
    tokenizer = Tokenizer(q_levels)

    model = Wavenet(dims=checkpoint.dims)
    model = model.to(device)
    model.load_state_dict(checkpoint.state_dict)

    with torch.inference_mode():
        sampled = model.sample(
            8, steps=22050, mulaw=mulaw, tokenizer=tokenizer, verbose=True
        )


if __name__ == "__main__":
    sample("/tmp/test/checkpoint.best.pt", None)
