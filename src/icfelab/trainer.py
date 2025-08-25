"""Module for training related functions and the lightning module for the training setting."""
import math
from typing import Tuple, List

import lightning
import torch
from torch import optim, Tensor
from torch.nn import ConstantPad1d, TransformerEncoder
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

from icfelab.model import Normalizer


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

    lengths = get_length(input_indices)  # type: ignore
    max_length = torch.max(lengths)
    padded_input_indices = pad_data(input_indices, max_length)  # type: ignore
    padded_values = pad_data(values, max_length)  # type: ignore
    padding_mask = torch.arange(padded_values.shape[1])[None, :] < lengths[:, None]

    return padded_input_indices[:, :, None], padded_values[:, :, None], torch.stack(target)[:, :, None], padding_mask[:,
    :,
    None]  # type: ignore


def pad_data(data: List[torch.Tensor], length: int) \
        -> Tensor:
    """
    Pad crops and targets respectively to match the largest width/length and enable stacking them to an input and
    target tensor with all data of this batch.
    """
    padded_data = []
    for sequence in data:  # TODO: do this loop in C or with torch?
        if len(sequence) < length:
            transform = ConstantPad1d((0, length - len(sequence)), -0.5)
            padded_data.append(transform(sequence))
        else:
            padded_data.append(sequence)
    return torch.stack(padded_data)


def get_length(data: List[Tensor]) -> Tensor:
    """
    Determine maximum sequence length.
    """
    result = []
    for sequence in data:  # TODO: do this loop in C
        result.append(len(sequence))
    return torch.tensor(result)


class TransformerTrainer(lightning.LightningModule):
    """Lightning module for image recognition training. Predict step returns a source object from the dataset as well as
    the softmax prediction."""

    def __init__(self, model: TransformerEncoder, hyper_parameters: dict, real_data: bool) -> None:
        super().__init__()
        self.model = model
        self.model.real_data = real_data
        self.batch_size = hyper_parameters["batch_size"]
        self.hyper_parameters = hyper_parameters
        self.gaussian = model.gaussian

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> torch.Tensor:
        self.model.train()
        loss, secondary_loss, _ = self.run_model(*batch)
        if not self.gaussian:
            secondary_loss = loss
        self.log("train_loss", secondary_loss.detach().cpu(), batch_size=self.batch_size,
                 prog_bar=False, on_epoch=True, on_step=True)
        self.log("train_nll_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=True, on_epoch=True,
                 on_step=True)
        return loss

    def run_model(self, indices: Tensor, values: Tensor, target: Tensor, mask: Tensor) -> Tuple[
        Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Predict input image and calculate loss. The target is modified, so that it consists out of start token for
        the same length as the encoder result. Only after the encoder results have been processed, the actual output
        starts.
        """
        self.model.device = self.device  # type: ignore
        self.model.normalizer = Normalizer(values, mask, self.hyper_parameters["padding_value"])

        indices = indices.to(self.device)
        values = values.to(self.device)
        target = target.to(self.device)

        pred_tuple = self.model(indices, values, torch.arange(target.shape[-2]).float())
        loss, secondary_loss = self.calculate_loss(pred_tuple, target)

        return loss, secondary_loss, pred_tuple

    def calculate_loss(self, pred_tuple: Tensor, target: Tensor) -> Tuple[Tensor]:
        """
        Calculate gaussian nll loss if gaussian mode and rmse loss otherwise.
        Args:
            pred_tuple: if gaussian mode is activated the tuple contains mean and variance prediction.
            Otherwise the first element contains the desired output value.
            target: target values [B,L,1]
        Returns: main loss value, as well as secondary loss for validation in gaussian training
        """
        loss = self.calculate_rmse(pred_tuple[0], target)
        secondary_loss = torch.zeros_like(loss)
        if self.gaussian:
            secondary_loss = loss
            mean_pred, var_log_pred = pred_tuple
            target = self.model.normalizer(target, padding=False)
            loss = 0.5 * (math.log(2 * math.pi) + var_log_pred + (
                    (target[:, :, 0] - mean_pred) ** 2) / torch.exp(var_log_pred) + 1e-6)
            loss = torch.mean(loss)

            std = torch.mean(self.model.normalizer.difference * torch.sqrt(torch.exp(var_log_pred)))  # log mean std
            print(std.detach().cpu().item())

        return loss, secondary_loss

    def calculate_rmse(self, pred: Tensor, target: Tensor) -> Tensor:
        """Unnormalize prediction and compare to target data."""
        pred = self.model.normalizer.unnormalize(pred)
        loss = torch.sqrt(mse_loss(pred, target[:, :, 0]))
        return loss

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
        result = self.run_model(*batch)
        loss, secondary_loss, _ = result
        if self.gaussian:
            secondary_loss, loss, _ = result
        self.log(f"{name}_loss", loss.detach().cpu(), batch_size=self.batch_size, prog_bar=False)
        self.log(f"{name}_nll_loss", secondary_loss.detach().cpu(), batch_size=self.batch_size, prog_bar=False)

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[
        Tensor, Tuple[Tensor, Tensor, Tensor], Tensor]:
        """Evaluate test dataset"""
        self.model.eval()
        _, _, pred_tuple = self.run_model(*batch)

        if self.gaussian:
            mean_pred, var_pred = pred_tuple
            pred = self.model.normalizer.unnormalize(mean_pred)
            # std = torch.sqrt(torch.exp(var_pred))
            std = self.model.normalizer.difference * torch.sqrt(torch.exp(var_pred))
        else:
            pred = pred_tuple[0]
            pred = self.model.normalizer.unnormalize(pred)
            std = torch.zero_like(pred)

        return pred, batch, std

    def configure_optimizers(self) -> Optimizer:
        """Configure AdamW optimizer from config."""
        optimizer = optim.AdamW(self.parameters(), lr=self.hyper_parameters["learning_rate"],
                                weight_decay=self.hyper_parameters["weight_decay"])
        return optimizer
