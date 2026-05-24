"""
vram_aware_loader.py
--------------------
VRAM-aware model loading for VISIONE analysis services.

Loading strategy (in priority order):
  1. Direct CUDA load  — if free VRAM ≥ estimated model size × safety margin
  2. CPU RAM offload   — model kept in system RAM, inference on CPU
  3. SSD / disk offload — model weight shards written to VISIONE_OFFLOAD_DIR
                          (mapped under /cache/model_offload by default),
                          with accelerate dispatching individual layers across
                          CPU and disk as needed

Environment variables (all optional, have sensible defaults):
  VISIONE_VRAM_THRESHOLD  float 0-1, fraction of free VRAM required to attempt
                          GPU load (default: 0.85)
  VISIONE_OFFLOAD_MODE    "auto" | "cpu" | "disk"
                          "auto"  → tries GPU, falls back to CPU, then disk
                          "cpu"   → skip GPU entirely, use CPU only
                          "disk"  → force disk offload from the start
                          (default: "auto")
  VISIONE_OFFLOAD_DIR     absolute path inside the container where weight
                          shards are written during disk offload
                          (default: /cache/model_offload)

Public API
----------
  load_model_with_offload(load_fn, model_size_gb, offload_dir=None)
      Wraps any callable that returns a nn.Module. Returns the model placed on
      the best available device according to the strategy above.

  load_hf_model_with_offload(from_pretrained_fn, model_size_gb, offload_dir=None)
      Convenience wrapper for HuggingFace from_pretrained() callables.
      Passes device_map / offload_folder kwargs automatically.

  load_hub_model_with_offload(hub_load_fn, model_size_gb, offload_dir=None)
      Convenience wrapper for torch.hub.load() callables.
      Uses accelerate.dispatch_model for disk offloading.

  get_offload_dir()
      Returns the effective offload directory (env var or default).

  suggest_batch_size(requested, strategy)
      Returns a (possibly reduced) batch size appropriate for the chosen device
      strategy: GPU keeps original; CPU halves it; disk forces 1.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & env-var helpers
# ---------------------------------------------------------------------------

_DEFAULT_OFFLOAD_DIR = "/cache/model_offload"
_DEFAULT_VRAM_THRESHOLD = 0.85
_DEFAULT_OFFLOAD_MODE = "auto"

# Bytes per GB
_GB = 1024 ** 3


def get_offload_dir() -> str:
    """Return the configured (or default) disk-offload directory."""
    return os.environ.get("VISIONE_OFFLOAD_DIR", _DEFAULT_OFFLOAD_DIR)


def _get_vram_threshold() -> float:
    try:
        t = float(os.environ.get("VISIONE_VRAM_THRESHOLD", _DEFAULT_VRAM_THRESHOLD))
        return max(0.0, min(1.0, t))
    except ValueError:
        return _DEFAULT_VRAM_THRESHOLD


def _get_offload_mode() -> str:
    return os.environ.get("VISIONE_OFFLOAD_MODE", _DEFAULT_OFFLOAD_MODE).lower()


# ---------------------------------------------------------------------------
# VRAM probe
# ---------------------------------------------------------------------------

def probe_vram(model_size_gb: float) -> str:
    """
    Probe available VRAM and decide on a loading strategy.

    Returns one of: "cuda", "cpu", "disk"
    """
    mode = _get_offload_mode()

    # Honour explicit forced modes
    if mode == "disk":
        log.info("[VRAMLoader] VISIONE_OFFLOAD_MODE=disk — forcing disk offload")
        return "disk"
    if mode == "cpu":
        log.info("[VRAMLoader] VISIONE_OFFLOAD_MODE=cpu — forcing CPU load")
        return "cpu"

    # mode == "auto"
    if not torch.cuda.is_available():
        log.info("[VRAMLoader] CUDA not available — using CPU")
        return "cpu"

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        log.warning(f"[VRAMLoader] Could not query VRAM ({exc}) — defaulting to CPU")
        return "cpu"

    free_gb = free_bytes / _GB
    total_gb = total_bytes / _GB
    free_fraction = free_bytes / total_bytes
    threshold = _get_vram_threshold()

    log.info(
        f"[VRAMLoader] VRAM: {free_gb:.2f}/{total_gb:.2f} GB free "
        f"({free_fraction*100:.1f}%), model needs ~{model_size_gb:.2f} GB, "
        f"threshold={threshold*100:.0f}%"
    )

    if free_fraction >= threshold and free_gb >= model_size_gb:
        log.info("[VRAMLoader] Sufficient VRAM → loading onto GPU")
        return "cuda"

    if free_gb >= model_size_gb * 0.5:
        # Some VRAM but below threshold — try GPU but be ready to fall back
        log.info(
            f"[VRAMLoader] VRAM marginal ({free_gb:.2f} GB free, "
            f"needs ~{model_size_gb:.2f} GB) → will attempt GPU with OOM guard"
        )
        return "cuda_risky"

    log.info(
        f"[VRAMLoader] Insufficient VRAM ({free_gb:.2f} GB free, "
        f"needs ~{model_size_gb:.2f} GB) → CPU first, then disk if needed"
    )
    return "cpu"


# ---------------------------------------------------------------------------
# Batch-size advisor
# ---------------------------------------------------------------------------

def suggest_batch_size(requested: int, strategy: str) -> int:
    """
    Return a batch size suitable for the chosen memory strategy.

      GPU  (cuda / cuda_risky) → keep requested
      CPU                      → halve (min 1)
      disk                     → 1

    Args:
        requested: the batch size the user configured
        strategy:  one of "cuda", "cuda_risky", "cpu", "disk"

    Returns:
        Adjusted batch size (int ≥ 1)
    """
    if strategy in ("cuda", "cuda_risky"):
        return requested
    if strategy == "cpu":
        adjusted = max(1, requested // 2)
        if adjusted != requested:
            log.info(
                f"[VRAMLoader] CPU mode: reducing batch size {requested} → {adjusted}"
            )
        return adjusted
    # disk
    if requested != 1:
        log.info(
            f"[VRAMLoader] Disk-offload mode: reducing batch size {requested} → 1"
        )
    return 1


# ---------------------------------------------------------------------------
# OOM recovery helper
# ---------------------------------------------------------------------------

def _oom_guard(fn: Callable, fallback: Callable):
    """Try fn(); on CUDA OOM, flush cache and call fallback()."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning(
            "[VRAMLoader] CUDA OutOfMemoryError — flushing cache and falling back"
        )
        return fallback()


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_model_with_offload(
    load_fn: Callable[[], nn.Module],
    model_size_gb: float,
    offload_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load a model returned by *load_fn* using the appropriate memory strategy.

    The function applies the following cascade:
      1. Probe VRAM → determine strategy
      2. If "cuda" or "cuda_risky": call load_fn() then .to("cuda"),
         wrapped in an OOM guard that retries on CPU
      3. If "cpu": call load_fn() → model already on CPU (default)
      4. If "disk": call load_fn() on CPU, then use accelerate.dispatch_model
         to shard layers to disk

    Args:
        load_fn:        Zero-argument callable returning an nn.Module.
                        The model should be constructed on CPU (no .to() call
                        inside load_fn).
        model_size_gb:  Estimated model size in GB (used for VRAM probe).
        offload_dir:    Directory for disk offloading. Defaults to
                        get_offload_dir().

    Returns:
        nn.Module placed on the appropriate device.
    """
    offload_dir = offload_dir or get_offload_dir()
    strategy = probe_vram(model_size_gb)

    if strategy in ("cuda", "cuda_risky"):
        def _try_gpu():
            model = load_fn()
            return model.to("cuda")

        def _fallback_cpu():
            log.warning("[VRAMLoader] Falling back to CPU load after OOM")
            model = load_fn()
            return model  # stays on CPU

        return _oom_guard(_try_gpu, _fallback_cpu)

    if strategy == "cpu":
        model = load_fn()
        return model  # already on CPU

    # strategy == "disk"
    return _disk_offload_generic(load_fn, offload_dir)


def _disk_offload_generic(load_fn: Callable[[], nn.Module], offload_dir: str) -> nn.Module:
    """
    Load model on CPU then shard it to disk using accelerate.dispatch_model.
    """
    _ensure_offload_dir(offload_dir)
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except ImportError:
        log.warning(
            "[VRAMLoader] 'accelerate' not installed — falling back to pure CPU load"
        )
        return load_fn()

    log.info(f"[VRAMLoader] Disk offloading to: {offload_dir}")
    model = load_fn()
    model.eval()

    # Keep a tiny amount of CUDA memory if available, otherwise CPU only
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        cuda_max = f"{int(free_bytes * 0.8 / _GB * 1024)}MiB"
        max_memory = {0: cuda_max, "cpu": "8GiB"}
    else:
        max_memory = {"cpu": "8GiB"}

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=_get_no_split_classes(model),
    )
    model = dispatch_model(model, device_map=device_map, offload_dir=offload_dir)
    log.info(f"[VRAMLoader] Model dispatched with device_map: {device_map}")
    return model


# ---------------------------------------------------------------------------
# HuggingFace-specific loader
# ---------------------------------------------------------------------------

def load_hf_model_with_offload(
    from_pretrained_fn: Callable[..., nn.Module],
    model_size_gb: float,
    offload_dir: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
) -> nn.Module:
    """
    Load a HuggingFace model (AutoModel, CLIPModel, etc.) with VRAM-aware
    placement.

    Uses HuggingFace Accelerate's native *device_map* / *offload_folder*
    arguments when available, falling back to manual dispatch for older
    versions.

    Args:
        from_pretrained_fn: Callable with signature (*args, **kwargs) that
                            returns an nn.Module. Typically a lambda wrapping
                            ``SomeModel.from_pretrained(handle, ...)``.
                            Do NOT include device_map or offload_folder in the
                            lambda — they are injected here.
        model_size_gb:      Estimated model size in GB.
        offload_dir:        Override for the disk-offload directory.
        extra_kwargs:       Additional keyword arguments forwarded to
                            from_pretrained_fn (e.g. torch_dtype, cache_dir).

    Returns:
        nn.Module on the best available device.
    """
    offload_dir = offload_dir or get_offload_dir()
    extra_kwargs = extra_kwargs or {}
    strategy = probe_vram(model_size_gb)

    if strategy in ("cuda", "cuda_risky"):
        def _try_gpu():
            model = from_pretrained_fn(
                device_map={"": 0},
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )
            return model

        def _fallback_cpu():
            log.warning("[VRAMLoader] HF: OOM — falling back to CPU")
            return from_pretrained_fn(
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )

        return _oom_guard(_try_gpu, _fallback_cpu)

    if strategy == "cpu":
        return from_pretrained_fn(
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )

    # disk
    _ensure_offload_dir(offload_dir)
    log.info(f"[VRAMLoader] HF disk offload → {offload_dir}")
    return from_pretrained_fn(
        device_map="auto",
        offload_folder=offload_dir,
        offload_state_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# torch.hub-specific loader (uses accelerate.dispatch_model)
# ---------------------------------------------------------------------------

def load_hub_model_with_offload(
    hub_load_fn: Callable[[], nn.Module],
    model_size_gb: float,
    offload_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load a torch.hub model with VRAM-aware placement.

    For GPU and CPU strategies the model is simply moved with .to(device).
    For disk offloading, accelerate.dispatch_model is used to shard layers
    across CPU and the offload directory on SSD.

    Args:
        hub_load_fn:    Zero-argument callable returning an nn.Module loaded
                        on CPU (e.g. ``lambda: torch.hub.load(...)``).
        model_size_gb:  Estimated model size in GB.
        offload_dir:    Override for the disk-offload directory.

    Returns:
        nn.Module on the best available device.
    """
    offload_dir = offload_dir or get_offload_dir()
    strategy = probe_vram(model_size_gb)

    if strategy in ("cuda", "cuda_risky"):
        def _try_gpu():
            model = hub_load_fn()
            return model.to("cuda")

        def _fallback_cpu():
            log.warning("[VRAMLoader] Hub: OOM — falling back to CPU")
            return hub_load_fn()  # CPU

        return _oom_guard(_try_gpu, _fallback_cpu)

    if strategy == "cpu":
        model = hub_load_fn()
        return model  # already on CPU

    # disk
    return _disk_offload_generic(hub_load_fn, offload_dir)


# ---------------------------------------------------------------------------
# Checkpoint helpers (SSD-friendly)
# ---------------------------------------------------------------------------

def load_checkpoint_to_device(
    checkpoint_path: str,
    model: nn.Module,
    strategy: str,
    offload_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load a raw PyTorch checkpoint (.pt / .pth) into *model*, respecting the
    memory strategy.

    When strategy is "disk", the checkpoint is first mapped to CPU (avoiding
    a full VRAM spike) and then accelerate dispatches layers.

    Args:
        checkpoint_path:  Path to the .pt or .pth checkpoint file.
        model:            An already-instantiated nn.Module (on CPU).
        strategy:         The strategy string from probe_vram().
        offload_dir:      Disk offload directory (for "disk" strategy).

    Returns:
        The model with weights loaded, placed on the appropriate device.
    """
    offload_dir = offload_dir or get_offload_dir()

    log.info(f"[VRAMLoader] Loading checkpoint: {checkpoint_path} (strategy={strategy})")

    # Always map to CPU first to avoid unnecessary VRAM spike during load
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Allow checkpoint to be a plain state_dict or a dict containing one
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)

    if strategy in ("cuda", "cuda_risky"):
        def _try_gpu():
            return model.cuda()

        def _fallback_cpu():
            log.warning("[VRAMLoader] Checkpoint OOM — keeping model on CPU")
            return model

        return _oom_guard(_try_gpu, _fallback_cpu)

    if strategy == "cpu":
        return model  # already on CPU

    # disk
    return _disk_offload_generic(lambda: model, offload_dir)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_offload_dir(offload_dir: str) -> None:
    """Create the offload directory if it does not exist."""
    Path(offload_dir).mkdir(parents=True, exist_ok=True)


def _get_no_split_classes(model: nn.Module) -> list[str]:
    """
    Return a list of class names that should not be split across devices.
    Detects common transformer block types automatically.
    """
    known = [
        "CLIPEncoderLayer",
        "BertLayer",
        "ViTLayer",
        "T5Block",
        "OPTDecoderLayer",
        "LlamaDecoderLayer",
        "Block",          # used by some ViT implementations
    ]
    present = [cls.__name__ for cls in _iter_module_classes(model) if cls.__name__ in known]
    return list(set(present))


def _iter_module_classes(model: nn.Module):
    """Yield unique class objects for all submodules in *model*."""
    seen = set()
    for m in model.modules():
        cls = type(m)
        if cls not in seen:
            seen.add(cls)
            yield cls
