import numpy as np
import torch

from typing import Dict, Tuple, Optional, Union

from .clean import clean_image_palette, clean_image_strict_palette
from .nn import clean_image_nn


ImageArray = Union[np.ndarray, torch.Tensor]


def _to_numpy(image_array: ImageArray):
    if isinstance(image_array, np.ndarray):
        return image_array, False, None, None
    if isinstance(image_array, torch.Tensor):
        return (
            image_array.detach().cpu().numpy(),
            True,
            image_array.dtype,
            image_array.device,
        )
    raise TypeError("image_array must be a numpy.ndarray or torch.Tensor")


def _to_original_type(
    np_array: np.ndarray,
    is_torch: bool,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> ImageArray:
    if not is_torch:
        return np_array

    tensor = torch.from_numpy(np_array)

    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)

    return tensor


def clean_image(
    image_array: ImageArray,
    method: str,
    colour_map: Dict[int, Tuple[int, int, int]],
    *,
    model: Optional[object] = None,
    morph_kernel_size: int = 0,
    output_type: str = "rgb",
) -> ImageArray:
    """
    Clean a single image (numpy array or torch tensor) using the specified method.

    Args:
        image_array: Array/Tensor of shape (H, W, 3), dtype uint8.
        method: "palette", "pixel_decoder", or "strict_palette" to choose cleaning approach.
        model: Required when method="pixel_decoder". A trained model with forward(batch) returning class probabilities.
        colour_map: Required for all methods. Dict mapping class index -> (r,g,b).
        morph_kernel_size: Optional morphological clean kernel size (palette method only).
        output_type: "rgb" to return colour image, "index" to return integer mask.

    Returns:
        Cleaned image with the same container type as the input (np.ndarray or torch.Tensor):
        (H,W,3) uint8 when output_type="rgb", otherwise (H,W) uint8.
    """
    np_image, is_torch, orig_dtype, orig_device = _to_numpy(image_array)

    if np_image.ndim != 3 or np_image.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3)")

    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")

    if method == "palette":
        # Build palette ndarray from colour_map in index order and delegate to core function
        keys = sorted(colour_map.keys())
        palette = np.asarray([colour_map[k] for k in keys], dtype=np.uint8)

        cleaned = clean_image_palette(
            np_image,
            palette=palette,
            morph_kernel_size=morph_kernel_size,
            output_type=output_type,
        )

    elif method == "strict_palette":
        cleaned = clean_image_strict_palette(
            np_image,
            colour_map=colour_map,
            output_type=output_type,
        )

    elif method == "pixel_decoder":
        if model is None:
            raise ValueError("model must be provided for method='pixel_decoder'")

        cleaned = clean_image_nn(
            np_image,
            model=model,
            colour_map=colour_map,
            output_type=output_type,
        )

    else:
        raise ValueError(
            "method must be 'palette', 'strict_palette', or 'pixel_decoder'"
        )

    return _to_original_type(cleaned, is_torch, orig_dtype, orig_device)
