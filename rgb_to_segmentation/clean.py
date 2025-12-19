import os
from typing import List

import numpy as np

from PIL import Image
from scipy import ndimage


def parse_palette_from_string(colours_str: str) -> np.ndarray:
    parts = [p.strip() for p in colours_str.split(";") if p.strip()]
    pal = []
    for p in parts:
        rgb = tuple(int(x) for x in p.split(","))
        if len(rgb) != 3:
            raise ValueError(f"Invalid colour triple: {p}")
        pal.append(rgb)
    return np.asarray(pal, dtype=np.uint8)


def parse_palette_from_file(path: str) -> np.ndarray:
    pal = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rgb = tuple(int(x) for x in line.split(","))
            if len(rgb) != 3:
                raise ValueError(f"Invalid colour triple in file {path}: {line}")
            pal.append(rgb)
    if not pal:
        raise ValueError(f"No colours found in file: {path}")
    return np.asarray(pal, dtype=np.uint8)


def nearest_palette_image(image_array: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Assign each pixel in `image_array` (H,W,3 uint8) to the nearest colour in `palette` (K,3 uint8).
    Returns recoloured image array with same shape and dtype uint8.
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3)")

    h, w, _ = image_array.shape
    flat = image_array.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)

    # Compute squared distances between each pixel and each palette colour.
    # distances shape: (N_pixels, K)
    d = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d, axis=1)
    new_flat = pal[idx]
    new = new_flat.reshape(h, w, 3).astype(np.uint8)
    return new


def get_palette_for_image(image_array: np.ndarray, full_palette: np.ndarray) -> np.ndarray:
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


def process_file(input_path: str, output_path: str, palette: np.ndarray, kernel_size: int):
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {input_path}: cannot open image ({e})")
        return

    arr = np.array(img, dtype=np.uint8)
    
    # Reduce palette to only colours present in this image
    reduced_palette = get_palette_for_image(arr, palette)
    
    cleaned = nearest_palette_image(arr, reduced_palette)
    
    # Apply morphological transformations if kernel_size > 0
    if kernel_size > 0:
        cleaned = apply_morphological_clean(cleaned, kernel_size)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(cleaned).save(output_path)


def process_directory(
    input_dir: str, output_dir: str, palette: np.ndarray, exts: List[str], inplace: bool, name_filter: str = "", kernel_size: int = 0
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
            process_file(in_path, out_path, palette, kernel_size)


def clean_segmentation(input_dir: str, output_dir: str = None, inplace: bool = False, colours: str = None, colours_file: str = None, exts: str = ".png,.jpg,.jpeg,.tiff,.bmp,.gif", name_filter: str = "", morph_kernel_size: int = 3):
    """
    Clean segmentation images using palette-based color mapping.

    Args:
        input_dir (str): Path to input directory containing segmentation images.
        output_dir (str, optional): Directory where cleaned images will be written. Required if not inplace.
        inplace (bool): Overwrite input images in place.
        colours (str, optional): Semicolon-separated list of RGB triples.
        colours_file (str, optional): Path to a file listing RGB triples.
        exts (str): Comma-separated list of allowed image extensions.
        name_filter (str): Only process files whose name contains this substring.
        morph_kernel_size (int): Size of morphological kernel for boundary cleaning.
    """
    if not inplace and output_dir is None:
        raise ValueError("Either output_dir must be provided or inplace must be True")

    if colours_file:
        palette = parse_palette_from_file(colours_file)
    elif colours:
        palette = parse_palette_from_string(colours)
    else:
        raise ValueError("Either colours or colours_file must be provided")

    exts_list = [e if e.startswith(".") else "." + e for e in exts.split(",")]

    out_dir = output_dir if not inplace else input_dir

    if not inplace:
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Processing: input={input_dir} -> output={out_dir}, colours={len(palette)}, morph_kernel={morph_kernel_size}"
    )
    process_directory(input_dir, out_dir, palette, exts_list, inplace, name_filter, morph_kernel_size)
    print("Done.")