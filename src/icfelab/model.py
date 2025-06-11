"""Module for function estimation transformer model."""
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.transforms.functional import normalize


class FunctionEstimator(nn.Module):
    """Function estimation model consisting out of a TranformerEncoder, as well as a Linear Layer generating new function values.
    The Transformer creates a hidden representation of the input sequence. Input sequence has a time index and a
    function value.
    The output Linear Layer takes in a time index and the hidden representation of the input sequence to generate the
    function values at that time.
    The time index is in both cases projected by the same linear Layer.
    """

    def __init__(self, dim: int, num_head: int, num_layers: int, dim_feedforward: int) -> None:
        super().__init__()
        self.linear_indices = nn.Conv1d(1, dim // 2, 1)
        self.linear_value = nn.Conv1d(1, dim // 2, 1)
        self.bnorm = nn.BatchNorm1d(dim)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(dim, num_head, dim_feedforward), num_layers)
        self.instance_norm = nn.InstanceNorm1d(1)

        self.decoder = nn.Sequential(nn.Linear(dim // 2 + dim, (dim // 2 + dim) * 2, bias=True), nn.Tanh(),
                                     nn.Linear((dim // 2 + dim) * 2, (dim // 2 + dim) * 2, bias=True), nn.Tanh(),
                                     nn.Linear((dim // 2 + dim) * 2, (dim // 2 + dim) * 2, bias=True), nn.Tanh(),
                                     nn.Linear((dim // 2 + dim) * 2, 1, bias=True))
        self.device = "cpu"
        self.hidden = None

    def forward(self, input_indices: Tensor, values: Tensor, output_indices: Tensor) -> Tensor:
        """
        Args:
            input_indices: input time indices [B,1,L]
            values: function values corresponding to the input indices [B,1,L]
            output_indices: indices on which the function values should be computed [L,]
        Return:
            Estimated function values [B,1,L]
        """
        input_indices, output_indices, values = self.normalization(input_indices, output_indices, values)
        hidden = self.run_encoder(input_indices, values)
        result = []
        for i in range(output_indices.shape[-1]):
            output_index = torch.full((hidden.shape[0],), output_indices[..., i].item()).to(self.device)
            output_index = torch.squeeze(self.linear_indices(output_index[:, None, None]))
            result.append(self.decoder(torch.concat([hidden[..., 0], output_index], dim=-1)))
        return torch.hstack(result)

    def normalization(self, input_indices: Tensor, output_indices: Tensor, values: Tensor) -> Tuple[
        Tensor, Tensor, Tensor]:
        """
        Args:
            input_indices: input time indices [B,1,L]
            values: function values corresponding to the input indices [B,1,L]
            output_indices: indices on which the function values should be computed [L,]
        Normalize data.
        """
        input_indices = input_indices / len(output_indices)
        mean, std = torch.mean(values, dim=-1), torch.std(values, dim=-1)
        values = normalize(values, torch.squeeze(mean), torch.squeeze(std))
        output_indices = output_indices / len(output_indices)
        return input_indices, output_indices, values

    def run_encoder(self, input_indices: Tensor, values: Tensor) -> Tensor:
        """
        Args:
            input_indices: normalized input time indices [B,1,L]
            values: function normalized values corresponding to the input indices [B,1,L]

        Returns:
            hidden representation of the input sequence [B,1,L]
        """
        input_indices = self.linear_indices(input_indices)
        values = self.linear_value(values)
        input = torch.concat([input_indices, values], dim=-2)
        input = self.bnorm(input)
        input = torch.permute(input, (0, 2, 1))  # encoder uses [B,L,C]
        hidden = self.encoder(input)
        hidden = torch.permute(hidden, (0, 2, 1))
        return hidden

    def inference(self, output_index: Tensor) -> Tensor:
        """
        Args:
            output_index: normalized index on which the function values should be computed [B,1]
        Returns: function value [B,1]
        """
        assert self.hidden is not None, "Please run the encoder before inference."
        output_index = torch.full((self.hidden.shape[0],), output_index.item()).to(self.device)
        output_index = torch.squeeze(self.linear_indices(output_index[:, None, None]))
        return self.decoder(torch.concat([torch.squeeze(self.hidden[..., -1:], dim=-1), output_index], dim=-1))
