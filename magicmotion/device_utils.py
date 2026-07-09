"""Device selection helpers so MagicMotion runs on CUDA, Apple Silicon (MPS), or CPU.

Kept dependency-light (torch only) so it can be imported from the inference entry
points without pulling in the training stack (accelerate, cv2, ...).
"""

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Priority: explicit ``prefer`` arg > ``MAGICMOTION_DEVICE`` env var > CUDA > MPS > CPU.
    CUDA is preferred over MPS so behavior is unchanged on the NVIDIA machines the
    project was built for; only Macs (no CUDA) fall through to MPS.
    """
    override = prefer or os.environ.get("MAGICMOTION_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bfloat16(device: torch.device) -> bool:
    """Whether bf16 is a safe compute dtype on this device.

    CUDA: only on Ampere+ (bf16-capable) GPUs. MPS: yes on recent PyTorch. CPU: yes.
    """
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    return True  # mps (torch>=2.4) and cpu handle bf16
