import numpy as np
import torch


def sr_tensor_to_uint8_rgb(frames: torch.Tensor) -> np.ndarray:
    """Convert normalized NCHW RGB model output to NCHW uint8 RGB.

    Finite values outside [0, 1] are clipped at this image-output boundary.
    Values inside the range retain the pipeline's existing truncation semantics.
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be a torch.Tensor, got {type(frames).__name__}")
    if frames.ndim != 4:
        raise ValueError(f"frames must have NCHW layout, got shape {tuple(frames.shape)}")
    if frames.shape[1] != 3:
        raise ValueError(f"frames must have exactly 3 RGB channels, got {frames.shape[1]}")
    if frames.shape[0] == 0 or frames.shape[2] == 0 or frames.shape[3] == 0:
        raise ValueError(f"frames must have non-empty N, H, and W dimensions, got {tuple(frames.shape)}")
    if not frames.is_floating_point():
        raise TypeError(f"frames must have a floating-point dtype, got {frames.dtype}")
    bounds = torch.stack(torch.aminmax(frames.detach()))
    if not torch.isfinite(bounds).all().item():
        raise ValueError("frames contain NaN or infinite values")

    quantized = frames.detach().clamp(0, 1)
    quantized.mul_(255)
    return (
        quantized
        .to(dtype=torch.uint8, device="cpu")
        .contiguous()
        .numpy()
    )
