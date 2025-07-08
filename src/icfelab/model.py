"""Module for function estimation transformer model."""
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm


class Normalizer():
    """Normalization module, used for data normalizing and de-normalizing."""

    def __init__(self, data: Tensor, padding_mask: Tensor, padding_value: float) -> None:
        """
        Args:
            data: 1D sequence data [B,L,1]
            padding_mask: padding mask with 1 for data and 0 for padded values [B,L,1]
            padding_value: fixed value for padding
        """
        self.padding_value = padding_value
        self.is_padding = ~padding_mask  # invert padding mask so that padded values have the value 1.
        self.min_values, self.max_values = self.get_min_max(data)
        min_max_are_close = torch.isclose(self.min_values, self.max_values)
        self.difference = self.max_values - self.min_values
        self.difference[min_max_are_close] = 1

    def get_min_max(self, data) -> Tuple[Tensor, Tensor]:
        """Extract min and max values for each sequence. Pad values are substituted such that they are ignored during
        min a max operation.

        Args:
            data: 1D sequence data [B,L,1]"""
        min_look_up = self.substitute_padding_values(data, float("Inf"))
        max_look_up = self.substitute_padding_values(data, float("-Inf"))
        return torch.min(min_look_up, dim=-1)[0], torch.max(max_look_up, dim=-1)[0]  # type:ignore

    def substitute_padding_values(self, data: Tensor, value: float) -> Tensor:
        """Clone data and substitute padding values. This can be used to exclude padding values from min and max
        calculation.

        Args:
            data: 1D sequence data [B,L,1]"""
        result = data.clone()
        result[self.is_padding] = value
        return result

    def __call__(self, data: Tensor) -> Tensor:
        """
        Args:
            data: 1D sequence data [B,L,1]

        Returns:1D sequence data [B,L,1]
        """
        values = ((data[:, :, 0] - self.min_values) / (self.difference))[:, :, None]
        values[self.is_padding] = self.padding_value
        return values

    def unnormalize(self, data: Tensor) -> Tensor:
        """
        Reverse the normalization.
        Args:
            data: 1D sequence data [B,L,1]

        Returns: 1D sequence data [B,L,1]
        """
        return data * (self.difference) + self.min_values


class Decoder(nn.Module):
    """Feedforward decoder"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.linear_1 = nn.Linear(dim, dim * 2, bias=True)
        self.linear_2 = nn.Linear(dim * 2, dim * 2, bias=True)
        self.lnorm_1 = nn.LayerNorm(dim * 2, eps=1e-6)
        self.linear_3 = nn.Linear(dim * 2, dim * 2, bias=True)
        self.lnorm_2 = nn.LayerNorm(dim * 2, eps=1e-6)
        self.linear_4 = nn.Linear(dim * 2, 1, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        """Implement fowrward pass with residual connections."""

        hidden = self.linear_1(hidden)
        residual = hidden.clone()

        hidden = self.linear_2(hidden)
        hidden = self.lnorm_1(hidden)

        hidden = nn.Tanh()(hidden + residual)

        residual = hidden.clone()
        hidden = self.linear_2(hidden)
        hidden = self.lnorm_2(hidden)

        hidden = nn.Tanh()(hidden + residual)
        return self.linear_4(hidden)


def index_normalization(indices: Tensor, length: int) -> Tensor:
    """
    Normalize index data according to length.
    Args:
        indices: 1d sequence indices [B,L,1]
    """
    indices = indices / length
    return indices


class ZeroFeatureProjector():
    """Project Features onto a higher dimensional space, by filling missing values with zeros."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, data: Tensor, device: str) -> Tensor:
        """
        Args:
            data: 1d sequence [B,L,1]

        Returns: 1d sequence features [B,L,C]
        """
        features = torch.zeros((data.shape[0], data.shape[1], self.dim), device=device)
        features[:, :, 0] = data[:, :, 0]
        return features


class FunctionEstimator(nn.Module):
    """Function estimation model consisting out of a TranformerEncoder, as well as a Linear Layer generating new
    function values.
    The Transformer creates a hidden representation of the input sequence. Input sequence has a time index and a
    function value.
    The output Linear Layer takes in a time index and the hidden representation of the input sequence to generate the
    function values at that time.
    The time index is in both cases projected by the same linear Layer.
    """

    def __init__(self, dim: int, num_head: int, num_layers: int, dim_feedforward: int, gaussian: bool = False) -> None:
        super().__init__()
        self.gaussian = gaussian
        self.dim = dim

        # self.linear_indices = nn.Conv1d(1, dim // 2, 1)
        # self.linear_indices_2 = nn.Conv1d(1, dim // 2, 1)
        # self.linear_value = nn.Conv1d(1, dim // 2, 1)
        self.projector = ZeroFeatureProjector(dim // 2)
        self.normalizer = None  # has to be initialized with current batch

        encoder_norm = LayerNorm(dim)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(dim, num_head, dim_feedforward), num_layers,
                                          encoder_norm)
        self.decoder = Decoder(3 * self.projector.dim)
        if gaussian:
            self.var_decoder = Decoder(3 * self.projector.dim)

        self.device = "cpu"
        self.hidden = None

    def forward(self, input_indices: Tensor, values: Tensor, output_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_indices: input time indices [B,L,1]
            values: function values corresponding to the input indices [B,L,1]
            output_indices: indices on which the function values should be computed [L,]
        Return:
            Estimated function values [B,L,1]
        """
        input_indices = index_normalization(input_indices, len(output_indices))
        output_indices = index_normalization(output_indices, len(output_indices))
        # values = self.normalizer(values)
        hidden = self.run_encoder(input_indices, values)

        result = []
        result_var = []
        for i in range(output_indices.shape[-1]):
            output_index = torch.full((hidden.shape[0],), output_indices[i].item()).to(self.device)
            output_index = torch.squeeze(self.projector(output_index[:, None, None], self.device))
            decoder_output = self.decoder(torch.concat([hidden[:, 0, :], output_index], dim=-1))
            result.append(decoder_output)
            # result.append(test_value)
            # test_result.append(test_value_no_func)
            # if self.gaussian:
            # result_var.append(self.var_decoder(torch.concat([hidden[..., -1], output_index], dim=-1))) # todo: remove
        # if self.gaussian:
        #     return torch.hstack(result), torch.hstack(result_var)
        return torch.hstack(result), torch.ones((1, 1))

    def run_encoder(self, input_indices: Tensor, values: Tensor) -> Tensor:
        """
        Args:
            input_indices: normalized input time indices [B,L,1]
            values: function normalized values corresponding to the input indices [B,L,1]

        Returns:
            hidden representation of the input sequence [B,L,1]
        """
        # data = torch.permute(data, (0, 2, 1))
        # input_indices = self.linear_indices(input_indices)
        # values = self.linear_value(values)
        # hidden = torch.permute(hidden, (0, 2, 1))

        features = self.projector(values, self.device)
        input_indices_features = self.projector(input_indices, self.device)

        data = torch.concat([input_indices_features, features], dim=-1)

        hidden = self.encoder(data)
        return hidden

    # def inference(self, output_index: Tensor) -> Tensor:
    #     """
    #     Args:
    #         output_index: normalized index on which the function values should be computed [B,1]
    #     Returns: function value [B,1]
    #     """
    #     assert self.hidden is not None, "Please run the encoder before inference."
    #     output_index = torch.full((self.hidden.shape[0],), output_index.item()).to(self.device)
    #     output_index = torch.squeeze(self.linear_indices_2(output_index[:, None, None]))
    #     return self.decoder(torch.concat([torch.squeeze(self.hidden[..., -1:], dim=-1), output_index], dim=-1))
