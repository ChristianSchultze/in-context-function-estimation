from pathlib import Path
from typing import List

from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """Dataset class for sampled functions. Sample contains at least 10 datapoints."""

    def __init__(self, data: list, augmentation: bool = False):
        self.augmentation = augmentation
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]