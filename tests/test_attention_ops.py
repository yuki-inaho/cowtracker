"""Attention ops of the video transformer: FA3 only on Hopper, FA2 from Ampere on, cutlass before (Turing), and
exact attention for fp32 and on the CPU."""

import pytest
import torch
import xformers.ops as xops

from cowtracker.layers import video_transformer

FA2 = (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
FA3 = (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)
CUTLASS = (xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), FA3), ((12, 0), FA2), ((10, 0), FA2), ((8, 6), FA2), ((7, 5), CUTLASS), ((6, 1), CUTLASS)],
    ids=["hopper", "blackwell-consumer", "blackwell-datacenter", "ampere", "turing", "pascal"],
)
def test_flash_ops_selection_uses_fa3_only_on_hopper(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)

    assert video_transformer._get_flash_attention_ops() == expected


def test_flash_ops_selection_without_cuda_is_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert video_transformer._get_flash_attention_ops() is None


def test_cpu_attention_matches_softmax_attention():
    torch.manual_seed(0)
    attention = video_transformer.FlashAttention3(dim=64, num_heads=4, qkv_bias=True)
    x = torch.randn(2, 10, 64)
    batch, tokens, channels = x.shape
    q, k, v = attention.qkv(x).reshape(batch, tokens, 3, 4, 16).permute(2, 0, 3, 1, 4)  # each [B, heads, N, 16]
    weights = torch.softmax(q @ k.transpose(-2, -1) * attention.scale, dim=-1)
    expected = attention.proj((weights @ v).transpose(1, 2).reshape(batch, tokens, channels))

    with torch.no_grad():
        torch.testing.assert_close(attention(x), expected)
