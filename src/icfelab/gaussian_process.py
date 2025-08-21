import argparse
import json
import lzma
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from torch.nn.functional import mse_loss

from src.icfelab.utils import load_cfg, plot_gp


def predict(data: List[dict]) -> Tuple[list, list]:
    gp_results = []
    rbf_scales = []
    loss_values = []
    std_values = []
    std_std_values = []
    count = 0
    for function in data:
        if function["rbf_scale"] <= 0.1:
            continue
        count += 1
        # todo: normalize this as well?
        data = function["input"]
        rbf_scales.append(function["rbf_scale"])
        # rbf_kernel = ConstantKernel(constant_value=function["std"]**2) * RBF(length_scale=function["rbf_scale"])
        rbf_kernel = RBF(length_scale=function["rbf_scale"])
        # gp = GaussianProcessRegressor(kernel=rbf_kernel)
        gp = GaussianProcessRegressor(kernel=rbf_kernel, alpha=function["std"]**2)
        gp.fit(np.array(data["indices"])[:, None], np.array(data["values"])[:, None])
        gp_results.append(gp.predict(np.arange(cfg["grid_length"])[:, None], return_std=True))
        loss_values.append(torch.sqrt(
            mse_loss(torch.tensor(gp_results[-1][0]), torch.tensor(function["target"]))))  # todo do this all properly
        std_values.append(torch.mean(torch.tensor(gp_results[-1][1])))
        std_std_values.append(torch.std(torch.tensor(gp_results[-1][1])))
    print("GP RMSE", torch.tensor(loss_values).mean())
    print("GP STD MEAN", torch.tensor(std_values).mean())
    print("GP STD STD", torch.tensor(std_std_values).mean())
    print(count)
    return gp_results, rbf_scales


def plot_results(results: Tuple[list, list], data: List[dict], limit: int = 12):
    number = 0
    path = Path("data/" + args.name)
    for i, (gp_result, rbf_scale) in enumerate(zip(results[0], results[1])):
        if i > limit:
            continue
        plot_gp(torch.tensor(data[i]["target"]), torch.tensor(data[i]["input"]["indices"]),
                torch.tensor(data[i]["input"]["values"]), path / f"{number}",
                torch.tensor(gp_result[0]), torch.tensor(gp_result[1]), rbf_scale)
        number += 1


def main():
    with lzma.open(args.data_path, mode="rb") as file:
        data = json.loads(file.read().decode("utf-8"))
    results = predict(data)
    plot_results(results, data)


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train SSM OCR")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the model and the log files.",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=None,
        help="path for data file.",
    )
    parser.add_argument(
        "--config-path",
        "-cp",
        type=str,
        default="config/cfg.yml",
        help="Path to model config. When eval is active, this will be ignored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = load_cfg(args.config_path)
    torch.random.manual_seed(42)
    main()
