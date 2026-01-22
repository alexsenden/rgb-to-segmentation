import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_classifier import PixelClassifier


class PixelwiseClassifier(PixelClassifier):
    step = 0
    val_step = 0

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(output_dim=output_dim)
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return self.softmax(x)

    def image_to_batch(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).to(self.device)

    def batch_to_image(self, batch: torch.Tensor, height: int, width: int):
        return batch.reshape(1, height, width, 3).permute(0, 3, 1, 2)
