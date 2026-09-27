"""Runtime guards and provenance: explicit device, per-process VRAM limit, peak VRAM, git/torch metadata."""

import hashlib
import subprocess
from pathlib import Path

import torch

GB = 1024**3
REPO = Path(__file__).resolve().parents[2]


def require_device(name: str) -> torch.device:
    """``torch.device(name)``; asking for CUDA without CUDA is an error (no silent CPU fallback)."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu explicitly to run on the CPU")
    return device


def apply_vram_limit(limit_gb: float, device: int = 0) -> None:
    """Cap this process at ``limit_gb`` (allocations beyond it raise OOM) once that much VRAM is free."""
    free, total = torch.cuda.mem_get_info(device)
    if limit_gb * GB > total:
        raise ValueError(f"VRAM limit {limit_gb} GB exceeds the device total {total / GB:.2f} GB")
    if free < limit_gb * GB:
        raise RuntimeError(
            f"free VRAM {free / GB:.2f} GB is below the requested limit {limit_gb} GB; another process is using the GPU"
        )
    torch.cuda.set_per_process_memory_fraction(limit_gb * GB / total, device)


def enable_expandable_segments() -> None:
    """Let the CUDA caching allocator grow segments instead of fragmenting: with a hard cap, gigabytes of reserved but
    unused blocks otherwise end in OOM (seen on TAP-Vid DAVIS at 30 GB). Numerics are unaffected."""
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def peak_vram_gb() -> float | None:
    return torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else None


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_metadata() -> dict:
    """Code and library versions of this run (``git_commit`` is None outside a git checkout)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def checkpoint_info(path: str | Path | None) -> dict | None:
    """File name, size and sha256 of a checkpoint (None when no checkpoint file was used)."""
    if path is None:
        return None
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
