import torch
import pytest
from wavenet.data import MuLaw, Tokenizer


@pytest.mark.parametrize("mu", [2**i for i in [5, 8, 10, 12]])
def test_mulaw(mu: int):
    mulaw = MuLaw(mu)

    x = 2 * torch.rand(2, 1024) - 1
    xmulaw = mulaw(x)
    assert x.shape == xmulaw.shape
    xdecoded = mulaw.decode(xmulaw)
    assert torch.allclose(x, xdecoded)

    x = 2 * torch.rand(2, 1, 1024) - 1
    xmulaw = mulaw(x)
    assert x.shape == xmulaw.shape
    xdecoded = mulaw.decode(xmulaw)
    assert torch.allclose(x, xdecoded)


@pytest.mark.parametrize("mu", [2**i for i in [5, 8, 10, 12]])
def test_tokenizer(mu: int):
    mulaw = MuLaw(mu)
    tokenizer = Tokenizer(mu)

    x = 2 * torch.rand(2, 1024) - 1

    encoded = tokenizer.encode(mulaw(x))
    assert encoded.min() >= 0
    assert encoded.max() < mu

    assert torch.allclose(x, mulaw.decode(mulaw.encode(x)))
    assert torch.allclose(x, mulaw.encode(mulaw.decode(x)))

    encoded = torch.randint(0, tokenizer.boundaries.shape[-1], (2, 1024))
    assert torch.all(encoded == tokenizer.encode(tokenizer.decode(encoded)))
    assert torch.allclose(x, tokenizer.decode(tokenizer.encode(x)), atol=tokenizer.atol)
