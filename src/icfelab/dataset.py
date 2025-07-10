"""Minimal dataset module for handling function data."""
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """Dataset class for sampled functions. Sample contains at least 10 datapoints."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        function = self.data[idx]
        return (torch.tensor(function["input"]["indices"]).float(), torch.tensor(function["input"]["values"]),
                torch.tensor(function["target"]))
