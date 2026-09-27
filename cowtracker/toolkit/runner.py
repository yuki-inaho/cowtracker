"""CoWTracker inference on uint8 videos: size rule, full/windowed mode rule, and the checkpoint-backed tracker."""

import numpy as np
import torch

from cowtracker.toolkit.tracks import DenseTracks

# lcm of the ViT patch (14) and the ResNet side branch (downsamples by 16, then 3 transposed convs): other sizes
# fail with a patch-size assertion or a 63-vs-64 shape mismatch (measured on 330x550 ... 336x616).
SIZE_MULTIPLE = 112
MODES = ("auto", "full", "windowed")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def validate_size(size_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = size_hw
    if height <= 0 or width <= 0 or height % SIZE_MULTIPLE or width % SIZE_MULTIPLE:
        raise ValueError(
            f"inference size {height}x{width}: H and W must be positive multiples of {SIZE_MULTIPLE} "
            "(e.g. 336x560 for 16:9-ish video, 336x448 for 4:3)"
        )
    return size_hw


def select_mode(mode: str, frames: int, window_len: int) -> str:
    """``auto``: one full pass up to ``window_len`` frames, the upstream windowed model beyond."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode != "auto":
        return mode
    return "full" if frames <= window_len else "windowed"


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """``auto``: fp16 on CUDA (upstream demo), fp32 on the CPU; fp16/bf16 on the CPU are refused."""
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected auto or one of {list(DTYPES)}")
    if device.type == "cpu" and name != "fp32":
        raise ValueError(f"{name} inference on the CPU is not supported; use --dtype fp32 (or auto)")
    return DTYPES[name]


class CowTrackerRunner:
    """``DenseTracker`` backed by the released checkpoint (loaded once, used for both modes)."""

    def __init__(
        self, checkpoint: str | None, device: torch.device, window_len: int = 100, mode: str = "auto", dtype="auto"
    ):
        from cowtracker import CoWTrackerWindowed

        select_mode(mode, 1, window_len)
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.window_len = window_len
        self.mode = mode
        self.calls: list[dict] = []
        # cast on the CPU, then move: the fp32 checkpoint (2x the fp16 weights) never occupies the GPU
        self.model = CoWTrackerWindowed.from_checkpoint(
            checkpoint, window_len=window_len, stride=window_len, device="cpu", dtype=self.dtype
        ).to(device)
        self.attention = self._attention_ops()

    def _attention_ops(self) -> str:
        """Name of the xformers ops of the video transformer for this dtype and device (for provenance)."""
        from cowtracker.layers.video_transformer import CUTLASS_OPS, _get_flash_attention_ops

        if self.device.type != "cuda":
            return "torch_sdpa"
        ops = CUTLASS_OPS if self.dtype == torch.float32 else _get_flash_attention_ops()
        return ops[0].NAME

    @torch.no_grad()
    def __call__(self, video: np.ndarray) -> DenseTracks:
        mode = select_mode(self.mode, len(video), self.window_len)
        torch.cuda.empty_cache()  # suffixes of varying length fragment the cache (OOM on TAP-Vid DAVIS at 30 GB)
        frames = torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2).float().to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = (self.model.model if mode == "full" else self.model)(frames)
        self.calls.append({"frames": len(video), "mode": mode})
        return DenseTracks(*(out[key][0].float().cpu().numpy() for key in ("track", "vis", "conf")))


def build_runner(
    checkpoint: str | None,
    device_name: str,
    vram_limit_gb: float | None,
    window_len: int,
    mode: str,
    dtype: str = "auto",
) -> tuple[CowTrackerRunner, str]:
    """Runner on an explicit device under an optional VRAM limit, and the checkpoint file it loaded.

    Without ``checkpoint`` the released weights are fetched from the Hugging Face hub (upstream default).
    """
    from huggingface_hub import hf_hub_download

    from cowtracker.models.cowtracker import CoWTracker
    from cowtracker.toolkit import runtime

    device = runtime.require_device(device_name)
    if device.type == "cuda":
        runtime.enable_expandable_segments()
    if vram_limit_gb is not None:
        runtime.apply_vram_limit(vram_limit_gb, device.index or 0)
    if checkpoint is None:
        checkpoint = hf_hub_download(repo_id=CoWTracker.DEFAULT_REPO_ID, filename=CoWTracker.DEFAULT_FILENAME)
    return CowTrackerRunner(checkpoint, device, window_len=window_len, mode=mode, dtype=dtype), checkpoint
