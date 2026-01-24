import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def image_to_batch(self, x: torch.Tensor):
        """
        Convert image tensor to batch format for CNN processing.

        Args:
            x: Input tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Batch tensor of shape (B, C, H, W) or (B, 1, H, W) for targets
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x.to(self.device)
