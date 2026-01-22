import pytorch_lightning as pl
import torch
import torch.nn as nn


class PixelClassifier(pl.LightningModule):
    """
    Base class for pixel-wise classifiers.
    Provides common interface for different model architectures.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def image_to_batch(self, x: torch.Tensor):
        """Convert image tensor to batch for processing.

        Args:
            x: Input tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Batch tensor ready for model input
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Training step."""
        sample, target = batch
        probs = self(sample)
        loss = self.loss_fn(probs, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sample, target = batch
        probs = self(sample)
        loss = self.loss_fn(probs, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters())
