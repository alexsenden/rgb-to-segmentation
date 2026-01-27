import numpy as np
import torch

from typing import Dict, Tuple, Optional, Union

from .clean import clean_image_palette
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
        method: "palette", "pixel_decoder", or "cnn_decoder" to choose cleaning approach.
        model: Required when method="pixel_decoder" or "cnn_decoder". A trained model with forward(batch) returning class probabilities.
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

    elif method in ["pixel_decoder", "cnn_decoder"]:
        if model is None:
            raise ValueError(
                "model must be provided for method=['pixel_decoder', 'cnn_decoder']"
            )

        cleaned = clean_image_nn(
            np_image,
            model=model,
            colour_map=colour_map,
            output_type=output_type,
        )

    else:
        raise ValueError(
            "method must be 'palette', 'pixel_decoder', or 'cnn_decoder'"
        )

    return _to_original_type(cleaned, is_torch, orig_dtype, orig_device)

def convert_mask_format(
    image_array: np.ndarray,
    colour_map: Dict[int, Tuple[int, int, int]],
    output_type: str = "rgb",
) -> np.ndarray:
    """
    Convert between RGB and index mask formats using a colour map.
    
    Args:
        image_array: Either (H, W, 3) uint8 RGB image or (H, W) index mask
        colour_map: Dict mapping class index -> (r,g,b)
        output_type: "rgb" to return colour image, "index" to return integer mask
    
    Returns:
        np.ndarray: (H,W,3) uint8 if output_type='rgb'; else (H,W) uint8
    
    Raises:
        ValueError: If image_array has invalid shape or contains unmapped RGB values
    """
    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")
    
    # Check if input is RGB (H, W, 3) or index (H, W)
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        # Input is RGB, convert to index
        h, w, _ = image_array.shape
        flat_img = image_array.reshape(-1, 3)
        
        # Build reverse lookup: RGB tuple -> class index
        rgb_to_idx = {tuple(map(int, rgb)): idx for idx, rgb in colour_map.items()}
        
        # Find all unique RGB values in the image
        unique_colours = np.unique(flat_img, axis=0)
        
        # Check if all colours are in the colour_map
        unmapped_colours = []
        for colour in unique_colours:
            colour_tuple = tuple(map(int, colour))
            if colour_tuple not in rgb_to_idx:
                unmapped_colours.append(colour_tuple)
        
        if unmapped_colours:
            # Format error message with unmapped colours
            colour_strs = [f"RGB{c}" for c in unmapped_colours[:10]]  # Show first 10
            if len(unmapped_colours) > 10:
                colour_strs.append(f"... and {len(unmapped_colours) - 10} more")
            raise ValueError(
                f"Image contains {len(unmapped_colours)} RGB value(s) not in colour_map: "
                f"{', '.join(colour_strs)}. All pixel values must exactly match a colour in the map."
            )
        
        # Map each pixel to its class index
        flat_indices = np.array(
            [rgb_to_idx[tuple(map(int, px))] for px in flat_img], dtype=np.uint16
        )
        index_image = flat_indices.reshape(h, w).astype(np.uint8)
        
        if output_type == "index":
            return index_image
        else:
            # Convert back to RGB (roundtrip)
            rgb_output = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, rgb in colour_map.items():
                mask = index_image == idx
                rgb_output[mask] = rgb
            return rgb_output
    
    elif image_array.ndim == 2:
        # Input is index mask, convert to RGB or keep as index
        h, w = image_array.shape
        
        if output_type == "index":
            # Already in index format
            return image_array.astype(np.uint8)
        else:
            # Convert to RGB
            rgb_output = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, rgb in colour_map.items():
                mask = image_array == idx
                rgb_output[mask] = rgb
            return rgb_output
    
    else:
        raise ValueError(
            "image_array must have shape (H, W, 3) for RGB or (H, W) for index mask"
        )