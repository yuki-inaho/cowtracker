"""Attention-op selection of the video transformer: FlashAttention3 kernels exist only for Hopper (sm_90)."""

import pytest
import torch
import xformers.ops as xops

from cowtracker.layers import video_transformer

FA2 = (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
FA3 = (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), FA3), ((12, 0), FA2), ((10, 0), FA2), ((8, 6), FA2)],
    ids=["hopper", "blackwell-consumer", "blackwell-datacenter", "ampere"],
)
def test_flash_ops_selection_uses_fa3_only_on_hopper(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)

    assert video_transformer._get_flash_attention_ops() == expected


def test_flash_ops_selection_without_cuda_is_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert video_transformer._get_flash_attention_ops() is None
