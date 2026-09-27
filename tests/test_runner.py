"""Inference-size rule (H and W multiples of 112) and the full/windowed mode rule."""

import pytest

from cowtracker.toolkit.runner import select_mode, validate_size


@pytest.mark.parametrize("size", [(336, 560), (336, 448), (224, 336), (448, 560), (112, 112)])
def test_validate_size_accepts_multiples(size):
    assert validate_size(size) == size


@pytest.mark.parametrize("size", [(336, 504), (392, 560), (330, 550), (0, 112), (336, 616)])
def test_validate_size_rejects_others(size):
    with pytest.raises(ValueError, match="multiples of 112"):
        validate_size(size)


def test_select_mode_auto_uses_full_up_to_window():
    assert select_mode("auto", frames=100, window_len=100) == "full"
    assert select_mode("auto", frames=101, window_len=100) == "windowed"


def test_select_mode_explicit_choice_wins():
    assert select_mode("windowed", frames=8, window_len=100) == "windowed"
    assert select_mode("full", frames=250, window_len=100) == "full"


def test_select_mode_unknown_raises():
    with pytest.raises(ValueError, match="mode"):
        select_mode("sliding", frames=8, window_len=100)


def test_runner_releases_cached_blocks_before_each_call(monkeypatch):
    import numpy as np
    import torch

    from cowtracker.toolkit.runner import CowTrackerRunner

    events = []

    class FakeModel(torch.nn.Module):
        def forward(self, frames):
            events.append("model")
            shape = (1, *frames.shape[:1], *frames.shape[2:])
            return {"track": torch.zeros((*shape, 2)), "vis": torch.ones(shape), "conf": torch.ones(shape)}

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    runner = object.__new__(CowTrackerRunner)
    runner.device, runner.dtype, runner.window_len, runner.mode, runner.calls = (
        torch.device("cpu"),
        torch.float32,
        100,
        "auto",
        [],
    )
    runner.model = torch.nn.Module()
    runner.model.model = FakeModel()

    runner(np.zeros((3, 4, 6, 3), dtype=np.uint8))

    assert events == ["empty_cache", "model"]
