from typing import Any

import torch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def move_to_device(
    target: Any,
    active_device: torch.device,
    target_device: torch.device,
) -> torch.device:
    """Move a PyTorch model or pipeline to `target_device`, rolling back to CPU on failure."""
    if active_device == target_device:
        return active_device
    try:
        target.to(target_device)
    except RuntimeError:
        if target_device.type == "cuda":
            target.to("cpu")
            torch.cuda.empty_cache()
        raise
    return target_device


def offload_to_cpu(target: Any, active_device: torch.device) -> torch.device:
    """Move a CUDA-resident PyTorch model or pipeline back to CPU and clear cached VRAM."""
    if target is not None and active_device.type == "cuda":
        target.to("cpu")
        torch.cuda.empty_cache()
        return torch.device("cpu")
    return active_device


def prepare_model_for_inference(
    active_model: object,
    peer_models: tuple[object | None, ...],
    offload_inactive: bool = True,
) -> None:
    """Ensure peer models leave GPU memory before activating the requested model."""
    if offload_inactive:
        for peer in peer_models:
            if peer is None or peer is active_model:
                continue
            offload = getattr(peer, "offload_to_cpu", None)
            if callable(offload):
                offload()

    activate = getattr(active_model, "activate_device", None)
    if callable(activate):
        activate()
