import argparse
import json
import lzma
import operator
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from torchsummary import summary

from src.icfelab.dataset import SampleDataset
from src.icfelab.utils import run_processes, load_cfg, initialize_random_split


def run_multiple_gpus(args: argparse.Namespace) -> None:
    """Launch a process for each gpu."""
    processes = [Process(target=train, args=(args, i)) for i in range(args.gpus)]
    run_processes(
        processes
    )


def main() -> None:
    """Launch processes for each gpu if possible, otherwise call train directly."""
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")

    config_path = Path(args.config_path)
    print(f"Model config {config_path}")

    if torch.cuda.is_available() and args.gpus > 1:
        assert torch.cuda.device_count() >= args.gpus, (
            f"More gpus demanded than available! Demanded: "
            f"{args.gpus} Available: {torch.cuda.device_count()}"
        )
        run_multiple_gpus(args)
    else:
        train(args)


def train(args: argparse.Namespace, device_id: Optional[int] = None) -> None:
    """Initialize config, datasets and dataloader and run the lightning trainer."""
    torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    cfg = load_cfg(config_path)

    ckpt_dir = Path(f"models/{args.name}")

    device_id = device_id if device_id else 0

    data = load_lzma_json_data(data_path)

    indices, splits = initialize_random_split(len(data), cfg["dataset_ratio"])

    getter = operator.itemgetter(*indices[: splits[0]].tolist())
    train_dataset = SampleDataset(getter(data))

    getter = operator.itemgetter(*indices[splits[0] : splits[1]].tolist())
    validation_dataset = SampleDataset(getter(data))

    getter = operator.itemgetter(*indices[splits[1] :].tolist())
    test_dataset = SampleDataset(getter(data))


def load_lzma_json_data(data_path: Path) -> Any:
    """
    Load lzma compressed json data.
    """
    with lzma.open(data_path, mode="r", encoding='utf-8') as file:
        json_bytes = file.read()
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train SSM OCR")

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="number of epochs to train",
    )
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
    return parser.parse_args()

if __name__ == "__main__":
    main()
