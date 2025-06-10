"""Module for function estimation transformer model."""
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        self.encoder = TransformerEncoder(TransformerEncoderLayer(dim, num_head, dim_feedforward), num_layers)

        self.decoder = nn.Linear(dim//2 + dim, 1, bias=True)

    def forward(self, input_indices: Tensor, values: Tensor, output_indices: Tensor) -> Tensor:
        """
        Args:
            input_indices: input time indices [B,1,L]
            values: function values corresponding to the input indices [B,1,L]
            output_indices: indices on which the function values should be computed [B,1,L]
        Return:
            Estimated function values [B,1,L]
        """
        input_indices = self.linear_indices(input_indices)
        output_indices = self.linear_indices(output_indices)
        values = self.linear_value(values)
        input = torch.concat([input_indices, values], dim=-2)
        input = torch.permute(input, (0, 2, 1)) # encoder uses [B,L,C]
        hidden = self.encoder(input)
        hidden = torch.permute(hidden, (0, 2, 1))
        result = []
        for i in range(output_indices.shape[-1]):
            output_index = output_indices[..., i]
            result.append(self.decoder(torch.concat([torch.squeeze(hidden[..., -1:], dim=-1), output_index], dim=-1)))
        return torch.vstack(result)