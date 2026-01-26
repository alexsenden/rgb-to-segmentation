import torch
import torch.nn as nn


class PixelClassifier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        """Return the device where the model parameters live."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def image_to_batch(self, x: torch.Tensor):
        raise NotImplementedError

    def align_logits_and_target(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape predictions/targets so CrossEntropyLoss receives the expected shapes."""

        if logits.dim() == 4:
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            return logits, target

        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)
        return logits, target
