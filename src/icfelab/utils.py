"""Utility functions"""
import json
import lzma
from multiprocessing import Process
from pathlib import Path
from typing import List, Tuple, Any
import pandas as pd

import numpy as np
import matplot2tikz

import torch
import yaml
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.gaussian_process.kernels import RBF
# pylint: disable=no-name-in-module
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
    """Plot predicted function with target and context points."""
    # print(pred_data)
    # print(target_data[0])
    # print("next")
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

    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def plot_raw(indices: Tensor, values: Tensor,
             path: Path) -> None:
    """Plot ground truth function with context points."""
    # pylint: disable=duplicate-code
    plt.figure(figsize=(8, 4))

    plt.scatter(indices * 24, values, label="observations", color='green')

    plt.title("")
    plt.xlabel("hours")
    plt.ylabel("flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()  # pylint: disable=unused-variable
    matplot2tikz.clean_figure()
    matplot2tikz.save(path.with_suffix(".tex"))

    plt.savefig(path.with_suffix(".png"))
    plt.close()


def plot_test(target_data: Tensor, indices: Tensor, values: Tensor,
              path: Path) -> None:
    """Plot ground truth function with context points."""
    # pylint: disable=duplicate-code
    x_data = torch.arange(len(target_data)) / len(target_data)
    indices = indices / len(target_data)
    plt.figure(figsize=(8, 4))

    plt.plot(x_data, target_data, label="target", color='blue')
    plt.scatter(indices, values, label="context points", color='green')

    plt.title("")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    # fig = plt.gcf()
    # # pylint: disable=assignment-from-no-return
    # fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save(path / ".tex")

    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def plot_gp(target_data: Tensor, indices: Tensor, values: Tensor,
            path: Path, gp_data: Tensor, gp_std: Tensor, rbf_scale: float) -> None:
    """Plot ground truth function with context points."""
    # pylint: disable=duplicate-code
    x_data = torch.arange(len(target_data)) / len(target_data)
    indices = indices / len(target_data)
    plt.figure(figsize=(8, 4))

    plt.plot(x_data, gp_data, label="gp prediction", color='orange')
    plt.fill_between(x_data, gp_data - gp_std, gp_data + gp_std, color="tab:orange", alpha=0.3)
    plt.plot(x_data, target_data, label="target", color='blue')
    plt.scatter(indices, values, label="context points", color='green')

    plt.title(f"RBF-Scale: {round(rbf_scale, 2)}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()  # pylint: disable=unused-variable
    matplot2tikz.clean_figure()
    matplot2tikz.save(path.with_suffix(".tex"))

    plt.savefig(path.with_suffix(".png"))
    plt.close()


# pylint: disable=unused-argument
def plot_cepheid(pred_data: Tensor, pred_std: Tensor, target_data: Tensor, indices: Tensor, values: Tensor,
                 path: Path, rbf_scale: float) -> None:
    """Plot targets and context points with gp and model predictions."""
    # pylint: disable=duplicate-code
    indices = indices * 24
    x_data = np.linspace(indices[0], indices[-1], 128)  # cepheid normalisation
    plt.figure(figsize=(8, 4))

    plt.scatter(indices, values, label="context points", color='green')
    plt.plot(x_data, pred_data, label="prediction", color='red')
    plt.fill_between(x_data, pred_data - pred_std, pred_data + pred_std, color="tab:red", alpha=0.3)

    plt.xlabel("Hours")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf() # pylint: disable=unused-variable
    matplot2tikz.clean_figure()
    matplot2tikz.save(path.with_suffix(".tex"))

    plt.savefig(path.with_suffix(".png"))
    plt.close()


def plot_full_data(pred_data: Tensor, pred_std: Tensor, target_data: Tensor, indices: Tensor, values: Tensor,
                   path: Path, rbf_scale: float) -> None:
    """Plot targets and context points with gp and model predictions."""
    # pylint: disable=duplicate-code
    x_data = torch.arange(len(target_data)) / len(target_data)
    indices = indices / len(target_data)

    plt.figure(figsize=(8, 4))

    plt.plot(x_data, target_data, label="target", color='blue')
    plt.scatter(indices, values, label="context points", color='green')
    plt.plot(x_data, pred_data, label="prediction", color='red')
    plt.fill_between(x_data, pred_data - pred_std, pred_data + pred_std, color="tab:red", alpha=0.3)

    plt.title(f"RBF Scale: {round(rbf_scale, 2)}")
    plt.xlabel("hours")
    plt.ylabel("flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf() # pylint: disable=unused-variable
    matplot2tikz.clean_figure()
    matplot2tikz.save(path.with_suffix(".tex"))

    plt.savefig(path.with_suffix(".png"))
    plt.close()


def plot_target(target_data: Tensor, path: Path) -> None:
    """Plot ground truth function."""
    # pylint: disable=duplicate-code
    x_data = torch.arange(len(target_data)) / len(target_data)
    plt.figure(figsize=(8, 4))

    plt.plot(x_data, target_data, label="target", color='blue')

    plt.title("")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    # fig = plt.gcf()
    # # pylint: disable=assignment-from-no-return
    # fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save(path / ".tex")

    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    # pylint: disable=protected-access
    if hasattr(obj, "_ncols"):  # type: ignore
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def create_covariance(rbf_scale: float, grid_length: int = 128, interval: tuple = (0, 1)) -> Tuple[
    ndarray, RBF]:
    """Create a Gaussian process with RBF kernel."""
    x = np.linspace(interval[0], interval[1], grid_length).reshape(-1, 1)
    kernel = RBF(length_scale=rbf_scale)
    return kernel(x), kernel


def read_cepheid():
    """Read cepheid data from csv file and extracts barycentric julian data as well as normalized flux values.
    Columns: Serial Number, BJD, Raw flux, Zero-point shift, Scaling factor, normalized flux, magnitude.
    """
    filename = "cepheid_data.csv"

    column_names = ['COL1', 'COL2', 'COL3', 'COL4', 'COL5', 'COL6', 'COL7']
    df = pd.read_csv(filename,
                     delim_whitespace=True,
                     header=None,
                     names=column_names)

    selected_df = df[['COL2', 'COL6']]
    tensor = torch.tensor(selected_df.values, dtype=torch.float32)
    tensor[:, 0] = tensor[:, 0] - tensor[:, 0][0]

    return tensor
