import os
from typing import List

import numpy as np

from PIL import Image
from scipy import ndimage
import torch


def clean_image_palette(
    image_array: np.ndarray,
    palette: np.ndarray,
    morph_kernel_size: int = 0,
    output_type: str = "rgb",
) -> np.ndarray:
    """
    Clean a single RGB image using palette-based nearest-colour mapping.

    Args:
        image_array: (H, W, 3) uint8
        palette: (K, 3) uint8
        morph_kernel_size: kernel size for morphological cleaning (0 disables)
        output_type: 'rgb' to return colour image; 'index' to return integer mask

    Returns:
        np.ndarray: (H,W,3) uint8 if output_type='rgb'; else (H,W) uint8
    """
    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")

    # Reduce palette to colours present in the image for efficiency
    reduced_palette = get_palette_for_image(image_array, palette)

    # Map to nearest colours
    cleaned_rgb = nearest_palette_image(image_array, reduced_palette)

    # Optional morphological clean
    if morph_kernel_size > 0:
        cleaned_rgb = apply_morphological_clean(cleaned_rgb, morph_kernel_size)

    if output_type == "rgb":
        return cleaned_rgb
    else:
        return rgb_image_to_index(cleaned_rgb, reduced_palette).astype(np.uint8)


def clean_image_strict_palette(
    image_array: np.ndarray,
    colour_map: dict,
    output_type: str = "rgb",
) -> np.ndarray:
    """
    Strictly map RGB values to indices based on colour_map.
    Raises an error if any RGB value is not found in the colour_map.

    Args:
        image_array: (H, W, 3) uint8 numpy array
        colour_map: Dict mapping class index -> (r,g,b)
        output_type: 'rgb' to return colour image; 'index' to return integer mask

    Returns:
        np.ndarray: (H,W,3) uint8 if output_type='rgb'; else (H,W) uint8
    """
    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")

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
        # Convert back to RGB using colour_map
        rgb_output = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, rgb in colour_map.items():
            mask = index_image == idx
            rgb_output[mask] = rgb
        return rgb_output


def nearest_palette_image(image_array: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Assign each pixel in `image_array` (H,W,3 uint8) to the nearest colour in `palette` (K,3 uint8).
    Returns recoloured image array with same shape and dtype uint8.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3)")

    h, w, _ = image_array.shape
    flat = image_array.reshape(-1, 3).astype(np.int64)
    pal = palette.astype(np.int64)

    # Compute squared distances between each pixel and each palette colour.
    # distances shape: (N_pixels, K)
    d = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)

    idx = np.argmin(d, axis=1)
    new_flat = pal[idx]
    new = new_flat.reshape(h, w, 3).astype(np.uint8)
    return new


def get_palette_for_image(
    image_array: np.ndarray, full_palette: np.ndarray
) -> np.ndarray:
    """
    Identify which colours from the full palette are present in the image,
    and return only those colours.
    """
    h, w, _ = image_array.shape
    flat_img = image_array.reshape(-1, 3).astype(np.int16)
    pal = full_palette.astype(np.int16)

    # For each pixel, find the nearest palette colour
    d = np.sum((flat_img[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d, axis=1)

    # Get unique indices that are actually used
    unique_idx = np.unique(idx)

    # Return only the palette colours that are used
    return full_palette[unique_idx]


def apply_morphological_clean(image_array: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological closing (erosion followed by dilation) per class to clean up
    class boundaries and remove noise.
    """
    if kernel_size <= 0:
        return image_array

    # Create morphological kernel
    kernel = ndimage.generate_binary_structure(2, 2)

    # Get unique colours that actually appear in the image
    h, w, _ = image_array.shape
    flat_img = image_array.reshape(-1, 3)
    unique_colours = np.unique(flat_img, axis=0)

    # Process each class separately to avoid blending
    result = np.zeros_like(image_array)

    for colour in unique_colours:
        # Create binary mask for this class
        mask = np.all(image_array == colour, axis=-1)

        # Apply closing: erosion then dilation
        for _ in range(kernel_size):
            mask = ndimage.binary_erosion(mask, structure=kernel)
        for _ in range(kernel_size):
            mask = ndimage.binary_dilation(mask, structure=kernel)

        # Assign pixels back
        result[mask] = colour

    # Fill any remaining pixels (from eroded areas) with nearest colour from result
    unfilled = ~np.any(result != 0, axis=-1)
    if np.any(unfilled):
        # For unfilled pixels, use nearest palette colour again or copy from nearby
        result[unfilled] = image_array[unfilled]

    return result


def rgb_image_to_index(image_array: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Map each RGB pixel in `image_array` to the index of the matching colour in `palette`.
    Assumes pixels take values from `palette`.
    """
    h, w, _ = image_array.shape
    palette_list = [tuple(map(int, c)) for c in palette.tolist()]
    lookup = {c: i for i, c in enumerate(palette_list)}
    flat = image_array.reshape(-1, 3)
    idx = np.array([lookup[tuple(map(int, px))] for px in flat], dtype=np.uint16)
    return idx.reshape(h, w)


def process_file(
    input_path: str,
    output_path: str,
    palette: np.ndarray,
    kernel_size: int,
    output_type: str = "rgb",
):
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {input_path}: cannot open image ({e})")
        return

    arr = np.array(img, dtype=np.uint8)

    # Clean image using core function
    cleaned = clean_image_palette(
        arr, palette=palette, morph_kernel_size=kernel_size, output_type=output_type
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_type == "rgb":
        Image.fromarray(cleaned).save(output_path)
    elif output_type == "index":
        Image.fromarray(cleaned.astype(np.uint8), mode="L").save(output_path)
    else:
        raise ValueError("output_type must be 'rgb' or 'index'")


def process_directory(
    input_dir: str,
    output_dir: str,
    palette: np.ndarray,
    exts: List[str],
    inplace: bool,
    name_filter: str = "",
    kernel_size: int = 0,
    output_type: str = "rgb",
):
    exts = [e.lower().strip() for e in exts]
    for root, dirs, files in os.walk(input_dir):
        # Determine the corresponding output root
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel) if not inplace else root
        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not any(fname.lower().endswith(e) for e in exts):
                continue
            if name_filter and name_filter not in fname:
                continue
            in_path = os.path.join(root, fname)
            out_path = os.path.join(out_root, fname)
            process_file(in_path, out_path, palette, kernel_size, output_type)


def clean_segmentation(
    input_dir: str,
    output_dir: str = None,
    inplace: bool = False,
    palette: np.ndarray = None,
    exts: str = ".png,.jpg,.jpeg,.tiff,.bmp,.gif",
    name_filter: str = "",
    morph_kernel_size: int = 3,
    output_type: str = "rgb",
):
    """
    Clean segmentation images using palette-based color mapping.

    Args:
        input_dir (str): Path to input directory containing segmentation images.
        output_dir (str, optional): Directory where cleaned images will be written. Required if not inplace.
        inplace (bool): Overwrite input images in place.
        palette (np.ndarray): Array of RGB triples (K, 3) uint8.
        exts (str): Comma-separated list of allowed image extensions.
        name_filter (str): Only process files whose name contains this substring.
        morph_kernel_size (int): Size of morphological kernel for boundary cleaning.
    """
    if not inplace and output_dir is None:
        raise ValueError("Either output_dir must be provided or inplace must be True")

    if palette is None:
        raise ValueError("palette must be provided")

    exts_list = [e if e.startswith(".") else "." + e for e in exts.split(",")]

    out_dir = output_dir if not inplace else input_dir

    if not inplace:
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Processing: input={input_dir} -> output={out_dir}, colours={len(palette)}, morph_kernel={morph_kernel_size}, output_type={output_type}"
    )
    process_directory(
        input_dir,
        out_dir,
        palette,
        exts_list,
        inplace,
        name_filter,
        morph_kernel_size,
        output_type,
    )
    print("Done.")
