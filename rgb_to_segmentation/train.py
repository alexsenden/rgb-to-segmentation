import os
from typing import Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
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

        self.items = []
        for paired_files in paired_filenames:
            sample_path = paired_files["sample"]
            target_path = paired_files["target"]

            sample = (
                read_image(sample_path, mode="RGB") / 127.5
            ) - 1.0  # Normalize to [-1, 1]
            target = read_image(target_path, mode="RGB")
            target = map_colour_to_int(target, self.colour_map)

            self.items.append((sample, target.to(torch.long)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def get_png_basename_pairs(directory: str) -> list[str]:
    dir_listing = os.listdir(directory)
    basenames = [
        filename.split(".")[0] for filename in dir_listing if filename.endswith(".png")
    ]
    full_names = [filename for filename in dir_listing if filename.endswith(".png")]

    return zip(basenames, full_names)


def get_paired_filenames(
    image_dir: str,
    label_dir: str,
    noisy_basename_pairs: list[tuple[str, str]],
    label_basename_pairs: list[tuple[str, str]],
    training_label_basename_pairs: list[tuple[str, str]],
) -> list[dict]:
    training_paired_filenames = []
    val_paired_filenames = []

    training_label_basenames = [
        basename for basename, _ in training_label_basename_pairs
    ]

    for noisy_basename, noisy_fullname in noisy_basename_pairs:
        targets = [
            target_fullname
            for target_basename, target_fullname in label_basename_pairs
            if target_basename == noisy_basename
        ]

        if len(targets) > 1:
            raise Exception(
                f"Multiple target files exist for noisy file {noisy_fullname}: {targets}."
            )
        elif len(targets) < 1:
            print(
                f"WARNING: No target found for noisy file {noisy_fullname}. Discarding."
            )
        else:
            target_basename = targets[0]

            if target_basename in training_label_basenames:
                training_paired_filenames.append(
                    {
                        "sample": f"{image_dir}/{noisy_fullname}",
                        "target": f"{label_dir}/{target_fullname}",
                    }
                )
            else:
                val_paired_filenames.append(
                    {
                        "sample": f"{image_dir}/{noisy_fullname}",
                        "target": f"{label_dir}/{target_fullname}",
                    }
                )

    return training_paired_filenames, val_paired_filenames


def get_dataloaders(
    image_dir, label_dir, colour_map, model
) -> tuple[DataLoader, DataLoader]:
    noisy_basenames = get_png_basename_pairs(image_dir)
    label_basenames = get_png_basename_pairs(label_dir)

    train_labels, val_labels = split_train_val(label_basenames, VALIDATION_FRACTION)

    training_paired_filenames, val_paired_filenames = get_paired_filenames(
        image_dir,
        label_dir,
        noisy_basenames,
        label_basenames,
        train_labels,
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
    device: torch.device = None,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
):
    """
    Train a neural network model for segmentation cleaning.

    Args:
        image_dir (str): Path to directory containing noisy images.
        label_dir (str): Path to directory containing target RGB labels.
        output_dir (str): Directory where model weights will be saved.
        colour_map (dict): Mapping from class indices to RGB tuples.
        model_type (str): The type of model to train.
        device (torch.device, optional): Device to use for training.
        max_epochs (int): Maximum training epochs.
        patience (int): Early stopping patience.
        learning_rate (float): Optimizer learning rate.
    """
    device = resolve_device(device)

    model = get_model(model_type, colour_map).to(device)
    train_dataloader, val_dataloader = get_dataloaders(
        image_dir, label_dir, colour_map, model
    )

    os.makedirs(output_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = os.path.join(output_dir, f"{model_type}.ckpt")

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer)
        val_loss = validate_one_epoch(model, val_dataloader)

        print(
            f"Epoch {epoch + 1}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model_checkpoint(model, model_type, colour_map, best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)."
            )
            break

    print(
        f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch + 1}. Saved to {best_model_path}."
    )


def resolve_device(preferred: torch.device | str | None) -> torch.device:
    if preferred is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(preferred, str):
        return torch.device(preferred)
    return preferred


def split_train_val(
    label_basename_pairs: list[tuple[str, str]], val_fraction: float
) -> Tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    total = len(label_basename_pairs)
    if total < 2:
        return label_basename_pairs, []

    val_size = max(1, int(total * val_fraction))
    val_size = min(val_size, total - 1)

    permutation = torch.randperm(total).tolist()
    val_indices = set(permutation[:val_size])

    train_labels = [
        label_basename_pairs[i] for i in range(total) if i not in val_indices
    ]
    val_labels = [label_basename_pairs[i] for i in val_indices]
    return train_labels, val_labels


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    batches = 0

    for sample, target in dataloader:
        optimizer.zero_grad()

        sample_batch = model.image_to_batch(sample)
        target_batch = model.image_to_batch(target).long()

        logits = model(sample_batch)
        logits, target_batch = model.align_logits_and_target(logits, target_batch)
        loss = model.loss_fn(logits, target_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(1, batches)


@torch.no_grad()
def validate_one_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    batches = 0

    for sample, target in dataloader:
        sample_batch = model.image_to_batch(sample)
        target_batch = model.image_to_batch(target).long()

        logits = model(sample_batch)
        logits, target_batch = model.align_logits_and_target(logits, target_batch)
        loss = model.loss_fn(logits, target_batch)

        total_loss += loss.item()
        batches += 1

    return total_loss / max(1, batches)


def save_model_checkpoint(model, model_type, colour_map, path):
    if model_type == "pixel_decoder":
        model_kwargs = {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
        }
    elif model_type == "cnn_decoder":
        model_kwargs = {
            "input_channels": model.input_channels,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
        }
    else:
        model_kwargs = {}

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "colour_map": colour_map,
        },
        path,
    )
