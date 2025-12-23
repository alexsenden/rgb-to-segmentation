import numpy as np

from typing import Dict, Tuple, Optional

from .clean import clean_image_palette
from .nn import clean_image_nn




def clean_image(
    image_array: np.ndarray,
    method: str,
    *,
    model: Optional[object] = None,
    colour_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    morph_kernel_size: int = 0,
    output_type: str = "rgb",
) -> np.ndarray:
    """
    Clean a single image (numpy array) using the specified method.

    Args:
        image_array: Numpy array of shape (H, W, 3), dtype uint8.
        method: "palette" or "nn" to choose cleaning approach.
        model: Required when method="nn". A trained model with forward(batch) returning class probabilities.
        colour_map: Required for both methods. Dict mapping class index -> (r,g,b).
        morph_kernel_size: Optional morphological clean kernel size (palette method only).
        output_type: "rgb" to return colour image, "index" to return integer mask.

    Returns:
        Cleaned image as numpy array: (H,W,3) uint8 when output_type="rgb", otherwise (H,W) uint8.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3)")

    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")

    if method == "palette":
        if colour_map is None:
            raise ValueError("colour_map must be provided for method='palette'")

        # Build palette ndarray from colour_map in index order and delegate to core function
        keys = sorted(colour_map.keys())
        palette = np.asarray([colour_map[k] for k in keys], dtype=np.uint8)

        return clean_image_palette(
            image_array,
            palette=palette,
            morph_kernel_size=morph_kernel_size,
            output_type=output_type,
        )

    elif method == "nn":
        if model is None:
            raise ValueError("model must be provided for method='nn'")

        if colour_map is None:
            raise ValueError("colour_map must be provided for method='nn'")

        return clean_image_nn(
            image_array,
            model=model,
            colour_map=colour_map,
            output_type=output_type,
        )

    else:
        raise ValueError("method must be 'palette' or 'nn'")
