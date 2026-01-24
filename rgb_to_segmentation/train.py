import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

from .nn import PixelwiseClassifier, CNNDecoder


VALIDATION_FRACTION = 0.2


def map_colour_to_int(sample, colour_map):
    """Vectorized color to class index mapping."""
    _, H, W = sample.shape

    # Create lookup tensor: shape (num_classes, 3)
    num_classes = len(colour_map)
    color_array = torch.zeros((num_classes, 3), dtype=sample.dtype)
    for idx, rgb in colour_map.items():
        color_array[idx] = torch.tensor(rgb, dtype=sample.dtype)

    # Reshape for broadcasting: (3, H, W) -> (H, W, 3) -> (H*W, 3)
    sample_flat = sample.permute(1, 2, 0).reshape(-1, 3)

    # Compute distances to all colors: (H*W, num_classes)
    distances = torch.cdist(sample_flat.float(), color_array.float(), p=2)

    # Assign to nearest color
    image_array = distances.argmin(dim=1).reshape(1, H, W)

    return image_array


def collate_fn(batch):
    """Stack samples/targets; model converts to batch on device."""
    samples, targets = zip(*batch)
    samples = torch.stack(samples, dim=0)
    targets = torch.stack(targets, dim=0)
    return samples, targets


class SegMaskDataset(Dataset):
    def __init__(self, paired_filenames, colour_map, model):
        self.colour_map = colour_map

        items = []
        for paired_files in paired_filenames:
            sample_path = paired_files["sample"]
            target_path = paired_files["target"]

            sample = (
                read_image(sample_path, mode="RGB") / 127.5
            ) - 1.0  # Normalize to [-1, 1]
            target = read_image(target_path, mode="RGB")
            target = map_colour_to_int(target, self.colour_map)

            items.append((sample, target.to(torch.long)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def get_png_basenames(directory: str) -> list[str]:
    return [
        os.path.splitext(filename)[0]
        for filename in os.listdir(directory)
        if filename.endswith(".png")
    ]


def get_paired_filenames(
    image_dir: str,
    label_dir: str,
    noisy_basenames: list[str],
    label_basenames: list[str],
    training_label_basenames: list[str],
) -> list[dict]:
    training_paired_filenames = []
    val_paired_filenames = []

    for noisy_image_name in noisy_basenames:
        targets = [
            target_file
            for target_file in label_basenames
            if target_file in noisy_image_name
        ]

        if len(targets) > 1:
            raise Exception(
                f"Multiple target files exist for noisy file {noisy_image_name}: {targets}."
            )
        elif len(targets) < 1:
            print(
                f"WARNING: No target found for noisy file {noisy_image_name}. Discarding."
            )
        else:
            target_filename = targets[0]

            if target_filename in training_label_basenames:
                training_paired_filenames.append(
                    {
                        "sample": f"{image_dir}/{noisy_image_name}.png",
                        "target": f"{label_dir}/{target_filename}.png",
                    }
                )
            else:
                val_paired_filenames.append(
                    {
                        "sample": f"{image_dir}/{noisy_image_name}.png",
                        "target": f"{label_dir}/{target_filename}.png",
                    }
                )

    return training_paired_filenames, val_paired_filenames


def get_dataloaders(
    image_dir, label_dir, colour_map, model
) -> tuple[DataLoader, DataLoader]:
    noisy_basenames = get_png_basenames(image_dir)
    label_basenames = get_png_basenames(label_dir)

    training_label_basenames, val_label_basenames = random_split(
        label_basenames, [1 - VALIDATION_FRACTION, VALIDATION_FRACTION]
    )

    training_paired_filenames, val_paired_filenames = get_paired_filenames(
        image_dir,
        label_dir,
        noisy_basenames,
        label_basenames,
        training_label_basenames,
    )

    train_dataset = SegMaskDataset(training_paired_filenames, colour_map, model)
    val_dataset = SegMaskDataset(val_paired_filenames, colour_map, model)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader


def get_model(model_type, colour_map):
    num_classes = len(colour_map.keys())

    if model_type == "pixel_decoder":
        return PixelwiseClassifier(input_dim=3, hidden_dim=32, output_dim=num_classes)
    elif model_type == "cnn_decoder":
        return CNNDecoder(input_channels=3, hidden_dim=64, output_dim=num_classes)

    raise ValueError(f"Invalid model type: {model_type}")


def train_model(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    colour_map: dict,
    model_type: str = "pixel_decoder",
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
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=output_dir,
        filename=model_type,
    )
    trainer = Trainer(max_epochs=100, callbacks=[early_stop, checkpoint])

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
