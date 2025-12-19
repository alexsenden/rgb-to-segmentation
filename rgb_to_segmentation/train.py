import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

from . import utils
from .nn import PixelwiseClassifier


VALIDATION_FRACTION = 0.2


def parse_colour_map_from_string(colour_map_str: str):
    colours = utils.parse_colours_from_string(colour_map_str)
    return {i: rgb for i, rgb in enumerate(colours)}


def parse_colour_map_from_file(path: str):
    colours = utils.parse_colours_from_file(path)
    return {i: rgb for i, rgb in enumerate(colours)}


def map_colour_to_int(sample, colour_map):
    _, H, W = sample.shape
    image_array = torch.zeros((1, H, W))

    for idx, rgb in colour_map.items():
        mask = torch.all(sample == torch.tensor(rgb).view(3, 1, 1), dim=0)
        image_array[0][mask] = idx

    return image_array


class SegMaskDataset(Dataset):
    def __init__(self, paired_filenames, colour_map, model):
        self.paired_filenames = paired_filenames
        self.colour_map = colour_map
        self.to_batch = model.image_to_batch

    def __len__(self):
        return len(self.paired_filenames)

    def __getitem__(self, index):
        sample_path = self.paired_filenames[index]["sample"]
        target_path = self.paired_filenames[index]["target"]

        sample = (
            read_image(sample_path, mode="RGB") / 127.5
        ) - 1.0  # Normalize to [-1, 1]
        target = read_image(target_path, mode="RGB")
        target = map_colour_to_int(target, self.colour_map)
        sample = self.to_batch(sample)
        target = self.to_batch(target).to(torch.long)

        return (sample, target)


def get_colour_map(colour_map_str=None, colour_map_file=None):
    if colour_map_file:
        return parse_colour_map_from_file(colour_map_file)
    elif colour_map_str:
        return parse_colour_map_from_string(colour_map_str)
    else:
        raise ValueError("Either colour_map or colour_map_file must be provided")


def get_paired_filenames(image_dir, label_dir):
    paired = []
    for root, dirs, files in os.walk(image_dir):
        rel = os.path.relpath(root, image_dir)
        label_root = os.path.join(label_dir, rel)
        if not os.path.exists(label_root):
            continue
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            sample_path = os.path.join(root, fname)
            target_path = os.path.join(label_root, fname)
            if os.path.exists(target_path):
                paired.append({"sample": sample_path, "target": target_path})
    return paired


def get_dataloaders(image_dir, label_dir, colour_map, model, batch_size=4):
    paired_filenames = get_paired_filenames(image_dir, label_dir)
    dataset = SegMaskDataset(paired_filenames, colour_map, model)

    train_size = int((1 - VALIDATION_FRACTION) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def get_model(model_type, colour_map):
    num_classes = len(colour_map.keys())

    if model_type == "pixelwise":
        return PixelwiseClassifier(input_dim=3, hidden_dim=32, output_dim=num_classes)

    raise ValueError(f"Invalid model type: {model_type}")


def train_model(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    colour_map: dict,
    model_type: str = "pixelwise",
):
    """
    Train a neural network model for segmentation cleaning.

    Args:
        image_dir (str): Path to directory containing noisy images.
        label_dir (str): Path to directory containing target RGB labels.
        output_dir (str): Directory where model weights will be saved.
        colour_map (dict): Mapping from class indices to RGB tuples.
        model_type (str): The type of model to train.
    """
    model = get_model(model_type, colour_map)
    train_dataloader, val_dataloader = get_dataloaders(
        image_dir, label_dir, colour_map, model
    )

    os.makedirs(output_dir, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=True)
    checkpoint = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, dirpath=output_dir
    )
    trainer = Trainer(max_epochs=100, callbacks=[early_stop, checkpoint])

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
