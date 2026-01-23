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

    def _align_logits_and_target(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape predictions/targets so CrossEntropyLoss receives the expected shapes."""

        # Handle channel-first 4D logits from CNN decoders: (B, C, H, W)
        if logits.dim() == 4:
            # Targets may come in as (B, 1, H, W) – squeeze the class channel
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            return logits, target

        # Everything else (e.g. flattened pixel classifiers): collapse batch/pixel dims
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)
        return logits, target

    def training_step(self, batch, batch_idx):
        """Training step."""
        sample, target = batch
        # Apply model-specific batching on GPU
        sample = self.image_to_batch(sample)
        target = self.image_to_batch(target).long()
        
        logits = self(sample)
        logits, target = self._align_logits_and_target(logits, target)
        loss = self.loss_fn(logits, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sample, target = batch
        # Apply model-specific batching on GPU
        sample = self.image_to_batch(sample)
        target = self.image_to_batch(target).long()
        
        logits = self(sample)
        logits, target = self._align_logits_and_target(logits, target)
        loss = self.loss_fn(logits, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters())
