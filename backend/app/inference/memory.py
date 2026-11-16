import torch


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


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
