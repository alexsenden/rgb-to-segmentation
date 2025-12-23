import os

import numpy as np
import torch

from PIL import Image
from torchvision.io import read_image

from .models.pixelwise_classifier import PixelwiseClassifier


def clean_image_nn(
    image_array: np.ndarray,
    model: object,
    colour_map: dict,
    output_type: str = "rgb",
) -> np.ndarray:
    """
    Clean a single image using the neural network model.

    Args:
        image_array: Numpy array (H, W, 3) uint8
        model: Trained model with `image_to_batch` and `forward`
        colour_map: Dict[int, (r,g,b)] used for RGB output mapping
        output_type: 'rgb' for colour image, 'index' for class-index mask

    Returns:
        np.ndarray: (H,W,3) uint8 if output_type='rgb'; else (H,W) uint8
    """
    if output_type not in ("rgb", "index"):
        raise ValueError("output_type must be 'rgb' or 'index'")

    import torch

    img_t = torch.from_numpy(image_array).permute(2, 0, 1).float() / 127.5 - 1.0
    h, w = img_t.shape[1], img_t.shape[2]
    batch = model.image_to_batch(img_t)

    with torch.no_grad():
        probs = model(batch)
        predicted = torch.argmax(probs, dim=-1).reshape(h, w)

    if output_type == "rgb":
        rgb_image_t = map_int_to_rgb(predicted, colour_map)
        return rgb_image_t.permute(1, 2, 0).numpy()
    else:
        return predicted.cpu().numpy().astype(np.uint8)


def load_model(model_path: str):
    print(model_path)
    model = PixelwiseClassifier.load_from_checkpoint(checkpoint_path=model_path)
    model.eval()

    return model


def map_int_to_rgb(indexed_image: torch.Tensor, colour_map: dict):
    h, w = indexed_image.shape
    rgb_image = torch.zeros((3, h, w), dtype=torch.uint8)

    for idx, rgb in colour_map.items():
        mask = indexed_image == idx
        for c in range(3):
            rgb_image[c][mask] = rgb[c]

    return rgb_image


def run_inference(
    input_dir: str,
    output_dir: str,
    inplace: bool = False,
    model_path: str = None,
    colour_map: dict = None,
    exts: str = ".png,.jpg,.jpeg,.tiff,.bmp,.gif",
    name_filter: str = "",
    output_type: str = "rgb",
):
    """
    Run neural network inference on segmentation images.

    Args:
        input_dir (str): Path to input directory containing images.
        output_dir (str, optional): Directory where output images will be written.
        inplace (bool): Overwrite input images in place.
        model_path (str): Path to the trained model file.
        colour_map (dict): Mapping from class indices to RGB tuples.
        exts (str): Comma-separated list of allowed image extensions.
        name_filter (str): Only process files whose name contains this substring.
    """
    if not inplace and output_dir is None:
        raise ValueError("Either output_dir must be provided or inplace must be True")

    if model_path is None:
        raise ValueError("model_path must be provided")

    if colour_map is None:
        raise ValueError("colour_map must be provided")

    model = load_model(model_path)

    exts_list = [e.lower().strip() for e in exts.split(",")]
    out_dir = output_dir if not inplace else input_dir

    if not inplace:
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Running inference: input={input_dir} -> output={out_dir}, output_type={output_type}"
    )

    for root, dirs, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(out_dir, rel) if not inplace else root

        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not any(fname.lower().endswith(e) for e in exts_list):
                continue
            if name_filter and name_filter not in fname:
                continue

            in_path = os.path.join(root, fname)
            out_path = os.path.join(out_root, fname)

            try:
                # Load as numpy for the shared single-image cleaner
                pil_img = Image.open(in_path).convert("RGB")
                np_img = np.array(pil_img, dtype=np.uint8)

                cleaned = clean_image_nn(
                    np_img, model=model, colour_map=colour_map, output_type=output_type
                )

                if output_type == "rgb":
                    pil_image = Image.fromarray(cleaned)
                    pil_image.save(out_path)
                elif output_type == "index":
                    pil_image = Image.fromarray(cleaned.astype("uint8"), mode="L")
                    pil_image.save(out_path)
                else:
                    raise ValueError("output_type must be 'rgb' or 'index'")

            except Exception as e:
                print(f"Skipping {in_path}: {e}")

    print("Inference done.")
