import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_classifier import PixelClassifier


class CNNDecoder(PixelClassifier):
    """
    CNN-based pixel classifier for image segmentation.

    Uses a convolutional encoder-decoder architecture to process images
    and produce per-pixel classifications.
    """

    def __init__(self, input_channels=3, hidden_dim=64, output_dim=2):
        """
        Initialize CNN decoder.

        Args:
            input_channels: Number of input channels (e.g., 3 for RGB)
            hidden_dim: Number of hidden channels in conv layers
            output_dim: Number of output classes
        """
        super().__init__(output_dim=output_dim)
        self.save_hyperparameters()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        # Encoder: downsampling with convolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        # Decoder: upsampling with transpose convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_classes, H, W) with softmax applied
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.softmax(decoded)

    def image_to_batch(self, x: torch.Tensor):
        """
        Convert image tensor to batch format for CNN processing.

        Args:
            x: Input tensor of shape (C, H, W)

        Returns:
            Batch tensor of shape (1, C, H, W)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x.to(self.device)

    def training_step(self, batch, batch_idx):
        """Training step with softmax cross entropy loss."""
        sample, target = batch
        # sample shape: (B, C, H, W)
        # target shape: (B, H, W) with class indices

        probs = self(sample)  # (B, num_classes, H, W)

        # Compute loss
        loss = self.loss_fn(probs, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with softmax cross entropy loss."""
        sample, target = batch
        probs = self(sample)  # (B, num_classes, H, W)
        loss = self.loss_fn(probs, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss
