import pytest
import torch
from wavenet.wavenet import CausalConv, WavenetLayer, WavenetBlock, Wavenet


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
    cconv = CausalConv(
        in_channels=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    x = torch.randn(batch_size, dim, seq_len)
    # x.requires_grad_ = True
    x.requires_grad_()

    y = cconv(x)
    assert x.shape == y.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (y * mask).pow(2).mean().backward()

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
    # x.requires_grad_ = True
    x.requires_grad_()

    y, ys = wnlayer(x)
    assert x.shape == y.shape
    assert x.shape == ys.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (y * mask).pow(2).mean().backward()

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
    x.requires_grad_()

    y, ys = wnblock(x)
    assert x.shape == y.shape
    assert x.shape == ys.shape

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1).repeat(1, dim, 1)

    (y * mask).pow(2).mean().backward()

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
    wavenet = Wavenet(
        n_blocks=n_blocks,
        n_layers_per_block=n_layers_per_block,
        in_channels=1,
        dim=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    x = torch.randn(batch_size, 1, seq_len)
    x.requires_grad_()

    y = wavenet(x)
    assert x.shape[-1] == y.shape[-1]

    mask = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    l = torch.randint(1, seq_len - 1, size=(batch_size, 1))
    mask = mask < l
    mask = mask.unsqueeze(1)

    (y * mask).pow(2).mean().backward()

    assert x.grad is not None
    assert torch.all(x.grad[mask] != 0)
    assert torch.all(x.grad[~mask] == 0)


if __name__ == "__main__":
    test_causal_conv(8192, 7, 5, 2)
