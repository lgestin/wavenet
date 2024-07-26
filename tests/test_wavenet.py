import pytest
import torch
from wavenet.wavenet import (
    CausalConv1d,
    WavenetLayer,
    WavenetBlock,
    Wavenet,
    WavenetDims,
)


def init_weigths(module):
    if isinstance(module, CausalConv1d):
        module.weight.fill_(1)
        if module.bias is not None:
            module.bias.fill_(0)


@pytest.mark.parametrize(
    "batch_size,seq_len,dim,kernel_size,dilation",
    [
        (3, 8192, 7, 5, 2),
        (3, 2345, 7, 5, 2),
        (3, 8192, 7, 2, 2),
        (3, 8192, 7, 3, 2),
        (3, 8192, 7, 2, 3),
        (3, 8192, 7, 3, 3),
        (3, 8192, 7, 3, 5),
    ],
)
def test_causal_conv(
    batch_size: int,
    seq_len: int,
    dim: int,
    kernel_size: int,
    dilation: int,
):
    cconv = CausalConv1d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    x = torch.randn(batch_size, dim, seq_len)
    x.requires_grad = True

    y = cconv(x)
    assert x.shape == y.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (1e3 * y * mask).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.all(x.grad[mask] != 0)
    assert torch.all(x.grad[~mask] == 0)


@pytest.mark.parametrize(
    "batch_size,seq_len,dim,kernel_size,dilation",
    [
        (3, 8192, 7, 5, 2),
        (3, 2345, 7, 5, 2),
        (3, 8192, 7, 2, 2),
        (3, 8192, 7, 3, 2),
        (3, 8192, 7, 2, 3),
        (3, 8192, 7, 3, 3),
        (3, 8192, 7, 3, 5),
    ],
)
def test_wavenet_layer(
    batch_size: int,
    seq_len: int,
    dim: int,
    kernel_size: int,
    dilation: int,
):
    wnlayer = WavenetLayer(dim=dim, kernel_size=kernel_size, dilation=dilation)

    x = torch.randn(batch_size, dim, seq_len)
    x.requires_grad = True

    y, ys = wnlayer(x)
    assert x.shape == y.shape
    assert x.shape == ys.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (1e3 * y * mask).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.all(x.grad[mask] != 0)
    assert torch.all(x.grad[~mask] == 0)


@pytest.mark.parametrize(
    "batch_size,seq_len,n_layers,dim,kernel_size,dilation",
    [
        (3, 8192, 1, 7, 5, 2),
        (3, 8192, 2, 7, 5, 2),
        (3, 8192, 10, 7, 5, 2),
        (3, 2345, 2, 7, 5, 2),
        (3, 8192, 2, 7, 2, 2),
        (3, 8192, 2, 7, 3, 2),
        (3, 8192, 2, 7, 2, 3),
        (3, 8192, 2, 7, 3, 3),
        (3, 8192, 2, 7, 3, 5),
    ],
)
def test_wavenet_block(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    dim: int,
    kernel_size: int,
    dilation: int,
):
    wnblock = WavenetBlock(
        n_layers=n_layers,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    x = torch.randn(batch_size, dim, seq_len)
    x.requires_grad = True

    y, ys = wnblock(x)
    assert x.shape == y.shape
    assert x.shape == ys.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (1e3 * y * mask).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.all(x.grad[mask] != 0)
    assert torch.all(x.grad[~mask] == 0)


@pytest.mark.parametrize(
    "batch_size,seq_len,n_blocks,n_layers_per_block,dim,kernel_size,dilation",
    [
        (3, 8192, 1, 2, 7, 5, 2),
        (3, 8192, 2, 2, 7, 5, 2),
        (3, 8192, 10, 2, 7, 5, 2),
        (3, 2345, 2, 2, 7, 5, 2),
        (3, 8192, 2, 2, 7, 2, 2),
        (3, 8192, 2, 2, 7, 3, 2),
        (3, 8192, 2, 2, 7, 2, 3),
        (3, 8192, 2, 2, 7, 3, 3),
        (3, 8192, 2, 2, 7, 3, 5),
    ],
)
def test_wavenet(
    batch_size: int,
    seq_len: int,
    n_blocks: int,
    n_layers_per_block: int,
    dim: int,
    kernel_size: int,
    dilation: int,
):
    dims = WavenetDims(
        n_blocks=n_blocks,
        n_layers_per_block=n_layers_per_block,
        in_channels=1,
        dim=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    wavenet = Wavenet(dims=dims)

    x = torch.randn(batch_size, 1, seq_len)
    x.requires_grad = True

    y = wavenet(x)
    assert x.shape[-1] == y.shape[1]

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1)

    (1e3 * y * mask.transpose(1, 2)).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.all(x.grad[mask] != 0)
    assert torch.all(x.grad[~mask] == 0)


@pytest.mark.parametrize(
    "batch_size,seq_len,n_blocks,n_layers_per_block,dim,kernel_size,dilation",
    [(3, 128, 2, 2, 7, 5, 2)],
)
@torch.inference_mode()
def test_wavenet_sampling(
    batch_size: int,
    seq_len: int,
    n_blocks: int,
    n_layers_per_block: int,
    dim: int,
    kernel_size: int,
    dilation: int,
):
    dims = WavenetDims(
        n_blocks=n_blocks,
        n_layers_per_block=n_layers_per_block,
        in_channels=1,
        dim=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    wavenet = Wavenet(dims=dims)

    # sampled = wavenet.sample(n=batch_size, steps=seq_len)
    # assert sampled.shape[0] == batch_size
    # assert sampled.shape[-1] == seq_len


if __name__ == "__main__":
    test_wavenet(
        batch_size=3,
        seq_len=8192,
        n_blocks=2,
        n_layers_per_block=10,
        dim=7,
        kernel_size=2,
        dilation=2,
    )
