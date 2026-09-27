"""Runtime guards: explicit device selection, VRAM limit, run metadata."""

import pytest
import torch

from cowtracker.toolkit import runtime

GB = 1024**3


def test_require_cuda_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        runtime.require_device("cuda")


def test_require_cpu_is_accepted():
    assert runtime.require_device("cpu") == torch.device("cpu")


def test_vram_limit_rejects_when_free_memory_insufficient(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (10 * GB, 32 * GB))

    with pytest.raises(RuntimeError, match="free VRAM"):
        runtime.apply_vram_limit(16.0)


def test_vram_limit_sets_the_process_fraction(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (30 * GB, 32 * GB))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, *args: calls.append(fraction))

    runtime.apply_vram_limit(16.0)

    assert calls == [pytest.approx(0.5)]


def test_run_metadata_has_git_and_versions():
    metadata = runtime.run_metadata()

    assert set(metadata) >= {"git_commit", "git_dirty", "torch", "cuda"}
    assert metadata["torch"] == torch.__version__


def test_file_sha256(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")

    assert runtime.file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_enable_expandable_segments_sets_the_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)

    runtime.enable_expandable_segments()

    assert calls == ["expandable_segments:True"]
