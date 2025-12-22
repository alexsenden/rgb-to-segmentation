import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

from .nn import PixelwiseClassifier


VALIDATION_FRACTION = 0.2


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
        num_workers=2,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )

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
