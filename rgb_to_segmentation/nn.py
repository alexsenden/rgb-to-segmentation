import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.io import read_image


class PixelwiseClassifier(pl.LightningModule):
    step = 0
    val_step = 0

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return self.softmax(x)

    def training_step(self, batch, batch_idx):
        sample, target = batch
        probs = self(sample)
        target = F.one_hot(target.squeeze(-1), num_classes=self.output_dim).to(dtype=sample.dtype)
        loss = self.loss_fn(probs, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sample, target = batch
        probs = self(sample)
        target = F.one_hot(target.squeeze(-1), num_classes=self.output_dim).to(dtype=sample.dtype)
        loss = self.loss_fn(probs, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def image_to_batch(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def batch_to_image(self, batch: torch.Tensor, height: int, width: int):
        return batch.reshape(1, height, width, 3).permute(0, 3, 1, 2)


def load_model(model_path: str, input_dim: int, hidden_dim: int, output_dim: int):
    model = PixelwiseClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def parse_colour_map_from_string(colour_map_str: str):
    parts = [p.strip() for p in colour_map_str.split(";") if p.strip()]
    colour_map = {}
    for i, p in enumerate(parts):
        rgb = tuple(int(x) for x in p.split(","))
        if len(rgb) != 3:
            raise ValueError(f"Invalid colour triple: {p}")
        colour_map[i] = rgb
    return colour_map


def parse_colour_map_from_file(path: str):
    colour_map = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rgb = tuple(int(x) for x in line.split(","))
            if len(rgb) != 3:
                raise ValueError(f"Invalid colour triple in file {path}: {line}")
            colour_map[i] = rgb
    if not colour_map:
        raise ValueError(f"No colours found in file: {path}")
    return colour_map


def map_int_to_rgb(indexed_image: torch.Tensor, colour_map: dict):
    h, w = indexed_image.shape
    rgb_image = torch.zeros((3, h, w), dtype=torch.uint8)
    for idx, rgb in colour_map.items():
        mask = indexed_image == idx
        for c in range(3):
            rgb_image[c][mask] = rgb[c]
    return rgb_image


def run_inference(input_dir: str, output_dir: str, inplace: bool = False, model_path: str = None, colour_map: dict = None, exts: str = ".png,.jpg,.jpeg,.tiff,.bmp,.gif", name_filter: str = ""):
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

    num_classes = len(colour_map)
    model = load_model(model_path, input_dim=3, hidden_dim=32, output_dim=num_classes)

    exts_list = [e.lower().strip() for e in exts.split(",")]
    out_dir = output_dir if not inplace else input_dir

    if not inplace:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Running inference: input={input_dir} -> output={out_dir}")

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
                img = read_image(in_path, mode="RGB").float() / 127.5 - 1.0  # Normalize
                h, w = img.shape[1], img.shape[2]
                batch = model.image_to_batch(img)
                with torch.no_grad():
                    probs = model(batch)
                    predicted = torch.argmax(probs, dim=-1).reshape(h, w)
                rgb_image = map_int_to_rgb(predicted, colour_map)
                # Save as PIL Image
                pil_image = Image.fromarray(rgb_image.permute(1, 2, 0).numpy())
                pil_image.save(out_path)
            except Exception as e:
                print(f"Skipping {in_path}: {e}")

    print("Inference done.")