"""Module for training related functions and the lightning module for the training setting."""
from pathlib import Path
from typing import Tuple, List

import lightning
import torch
from torch import optim, Tensor
from torch.nn import ConstantPad1d, TransformerEncoder
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torchvision.transforms.functional import normalize

from icfelab.utils import plot_single_prediction, plot_test


def collate_fn(batch: Tuple[List[torch.Tensor], ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function, that pads crops horizontally to fit them all in one tensor batch.
    Args:
        batch: includes List[Tensor] data for input_indices, values, target
    Returns:
        input_indices, values, target with shape [B, 1, L]
    """
    input_indices, values, target = zip(*batch)
    assert len(input_indices) == len(values), (f"Invalid data. input indices(length {len(input_indices)}) and "
                                               f"values(length {len(values)}) must have the same length.")

    max_length = get_max_length(input_indices)  # type: ignore
    padded_input_indices = pad_data(input_indices, max_length)  # type: ignore
    padded_values = pad_data(values, max_length)  # type: ignore

    return torch.stack(padded_input_indices)[:, None, :], torch.stack(padded_values)[:, None, :], torch.stack(target)[:,
                                                                                                  None,
                                                                                                  :]  # type: ignore


def pad_data(data: List[torch.Tensor], length: int) \
        -> List[Tensor]:
    """
    Pad crops and targets respectively to match the largest width/length and enable stacking them to an input and
    target tensor with all data of this batch.
    """
    padded_data = []
    for sequence in data:
        if len(sequence) < length:
            transform = ConstantPad1d((0, length - len(sequence)), 0)
            padded_data.append(transform(sequence))
        else:
            padded_data.append(sequence)
    return padded_data


def get_max_length(data: List[Tensor]) -> Tuple[int, int]:
    """
    Determine maximum sequence length.
    """
    max_length = 0
    for sequence in data:
        length = len(sequence)
        max_length = max(max_length, length)
    return max_length


class TransformerTrainer(lightning.LightningModule):
    """Lightning module for image recognition training. Predict step returns a source object from the dataset as well as
    the softmax prediction."""

    def __init__(self, model: TransformerEncoder, hyper_parameters: dict) -> None:
        super().__init__()
        self.model = model
        self.batch_size = hyper_parameters["batch_size"]
        self.hyper_parameters = hyper_parameters

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> torch.Tensor:
        self.model.train()
        indices, values, target = batch
        loss, _ = self.run_model(indices, values, target)
        self.log("train_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=True, on_epoch=True,
                 on_step=True)
        return loss

    def run_model(self, indices: Tensor, values: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict input image and calculate loss. The target is modified, so that it consists out of start token for
        the same length as the encoder result. Only after the encoder results have been processed, the actual output
        starts.
        """
        self.model.device = self.device

        # indices = torch.squeeze(indices).cpu()
        # values = torch.squeeze(values).cpu()
        # target = torch.squeeze(target).cpu()
        #
        # number = 0
        # for i in range(len(indices)):
        #     plot_test(target[i], indices[i], values[i], Path(f"data/debug/{number}.png"))
        #     number += 1

        indices = indices.to(self.device)
        values = values.to(self.device)
        target = target.to(self.device)
        target = self.model.instance_norm(target)

        mean, std = torch.mean(values, dim=-1), torch.std(values, dim=-1)
        target = normalize(target, torch.squeeze(mean), torch.squeeze(std))
        pred = self.model(indices, values, torch.arange(target.shape[-1]).float())[None, None, :]
        loss = mse_loss(pred, target)
        return loss, pred

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Evaluate validation dataset"""
        self.model.eval()
        self.evaluate_prediction(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Evaluate test dataset"""
        self.model.eval()
        self.evaluate_prediction(batch, "test")

    def evaluate_prediction(self, batch: Tuple[Tensor, Tensor, Tensor], name: str) -> None:
        """Evaluate input batch and log with supplied name tag.
        Predicts model, converts output tokens to text and calculates levenshtein distance."""
        indices, values, target = batch
        loss, _ = self.run_model(indices, values, target)
        self.log(f"{name}_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=False)

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Evaluate test dataset"""
        self.model.eval()
        indices, values, target = batch
        mean, std = torch.mean(values, dim=-1), torch.std(values, dim=-1)
        _, pred = self.run_model(indices, values, target)
        target = normalize(target, torch.squeeze(mean), torch.squeeze(std))
        values = normalize(values, torch.squeeze(mean), torch.squeeze(std))
        batch = indices, values, target
        return pred, batch

    def configure_optimizers(self) -> Optimizer:
        """Configure AdamW optimizer from config."""
        optimizer = optim.AdamW(self.parameters(), lr=self.hyper_parameters["learning_rate"],
                                weight_decay=self.hyper_parameters["weight_decay"])
        return optimizer
