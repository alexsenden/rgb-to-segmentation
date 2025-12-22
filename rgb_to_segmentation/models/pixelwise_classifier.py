import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelwiseClassifier(pl.LightningModule):
    step = 0
    val_step = 0

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters()

        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return self.softmax(x)

    def training_step(self, batch, batch_idx):
        sample, target = batch
        probs = self(sample)
        target = F.one_hot(target.squeeze(-1), num_classes=self.output_dim).to(
            dtype=sample.dtype
        )
        loss = self.loss_fn(probs, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sample, target = batch
        probs = self(sample)
        target = F.one_hot(target.squeeze(-1), num_classes=self.output_dim).to(
            dtype=sample.dtype
        )
        loss = self.loss_fn(probs, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def image_to_batch(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).to(self.device)

    def batch_to_image(self, batch: torch.Tensor, height: int, width: int):
        return batch.reshape(1, height, width, 3).permute(0, 3, 1, 2)
