import json
import lzma
from multiprocessing import Process
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import randperm, Tensor
from tqdm import tqdm

def run_processes(
    processes: List[Process],
) -> None:
    """
    Launches basic processes.
    """
    for process in processes:
        process.start()
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()

def load_cfg(config_path: Path) -> dict:
    """Load yml config from supplied path."""
    with open(config_path, "r", encoding="utf-8") as file:
        cfg: dict = yaml.safe_load(file)
    return cfg

def initialize_random_split(
    size: int, ratio: Tuple[float, float, float]
) -> Tuple[list, Tuple[int, int]]:
    """
    Args:
        size(int): Dataset size
        ratio(list): Ratio for train, val and test dataset

    Returns:
        tuple: List of randomly selected indices, as well as int tuple with two values that indicate the split points
        between train and val, as well as between val and test.
    """
    assert sum(ratio) == 1, "ratio does not sum up to 1."
    assert len(ratio) == 3, "ratio does not have length 3"
    assert (
        int(ratio[0] * size) > 0
        and int(ratio[1] * size) > 0
        and int(ratio[2] * size) > 0
    ), (
        "Dataset is to small for given split ratios for test and validation dataset. "
        "Test or validation dataset have size of zero."
    )
    splits = int(ratio[0] * size), int(ratio[0] * size) + int(ratio[1] * size)
    indices = randperm(size, generator=torch.Generator().manual_seed(42)).tolist()
    return indices, splits


def load_lzma_json_data(data_path: Path) -> Any:
    """
    Load lzma compressed json data.
    """
    with lzma.open(data_path, mode="rb") as file:
        json_bytes = file.read()
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)


def plot_single_prediction(pred_data: Tensor, target_data: Tensor, indices: Tensor, values: Tensor, path: Path) -> None:
    x_data = torch.arange(len(pred_data)) / len(pred_data)
    indices = indices / len(pred_data)
    plt.figure(figsize=(8, 4))

    plt.plot(x_data, target_data, label="target", color='blue')
    plt.scatter(indices, values, label="context points", color='green')
    plt.plot(x_data, pred_data, label="prediction", color='red')

    plt.title("")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path)

def plot_test(target_data: Tensor, indices: Tensor, values: Tensor,
                           path: Path) -> None:
    x_data = torch.arange(len(target_data)) / len(target_data)
    indices = indices / len(target_data)
    plt.figure(figsize=(8, 4))

    plt.plot(x_data, target_data, label="target", color='blue')
    plt.scatter(indices, values, label="context points", color='green')
    # plt.plot(x_data, pred_data, label="prediction", color='red')

    plt.title("")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path)
