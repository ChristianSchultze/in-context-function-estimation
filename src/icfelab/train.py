import argparse
import operator
import os
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Union

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchinfo import summary

from icfelab.model import FunctionEstimator
from src.icfelab.dataset import SampleDataset
from src.icfelab.trainer import TransformerTrainer, collate_fn
from src.icfelab.utils import run_processes, load_cfg, initialize_random_split, load_lzma_json_data


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
        device_id = 0 if torch.cuda.is_available() else "cpu"
        train(args, device_id)


def train(args: argparse.Namespace, device_id: Union[int, str]) -> None:
    """Initialize config, datasets and dataloader and run the lightning trainer."""
    torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    cfg = load_cfg(config_path)

    ckpt_dir = Path(f"models/{args.name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = load_lzma_json_data(data_path)

    indices, splits = initialize_random_split(len(data), cfg["training"]["dataset_ratio"])

    getter = operator.itemgetter(*indices[: splits[0]])
    train_dataset = SampleDataset(getter(data))

    getter = operator.itemgetter(*indices[splits[0] : splits[1]])
    validation_dataset = SampleDataset(getter(data))

    getter = operator.itemgetter(*indices[splits[1] :])
    test_dataset = SampleDataset(getter(data))

    encoder_cfg = cfg["encoder"]
    model = FunctionEstimator(encoder_cfg["dim"], encoder_cfg["num_head"], encoder_cfg["num_layers"], encoder_cfg["dim_feedforward"])

    summary(model, input_size=((16, 1, 10),(16, 1, 10),(1, 1, 128)), device="cpu")
    batch_size = cfg["training"]["batch_size"]
    eval_batch_size = cfg["eval"]["batch_size"]

    lit_model = TransformerTrainer(model, cfg["training"])

    if not args.eval:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )
        assert len(train_loader) > 1, f"Train dataset is too small! Train batches: {len(train_loader)}"
        assert len(val_loader) > 1, f"Validation dataset is too small! Validation batches: {len(val_loader)}!"
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert len(test_loader) > 1, f"Test dataset is too small! Test batches: {len(test_loader)}"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename=f"{device_id}-{{epoch}}",
    )

    logger = TensorBoardLogger(f"logs/{args.name}", name=f"{device_id}")
    if device_id != "cpu":
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator="gpu",
            devices=[device_id],
            val_check_interval=0.5,
            limit_val_batches=0.5,
        )  # type: ignore
    else:
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator="cpu",
            val_check_interval=0.5,
            limit_val_batches=0.5,
        )  # type: ignore

    if args.eval:
        eval_path = Path(args.eval)
        model_path = (
            eval_path
            / [f for f in os.listdir(eval_path) if f.startswith(f"{device_id}")][0]
        )
        model = FunctionEstimator(encoder_cfg["dim"], encoder_cfg["num_head"], encoder_cfg["num_layers"], encoder_cfg["dim_feedforward"]).eval()
        lit_model = TransformerTrainer.load_from_checkpoint(
            model_path, model=model, hyper_parameters=cfg["training"]
        )
        trainer.test(lit_model, dataloaders=test_loader)
    else:
        # pylint: disable=possibly-used-before-assignment
        trainer.fit(
            model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        with open(ckpt_dir / "model.yml", "w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file)

        lit_model = TransformerTrainer.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=model,
            hyper_parameters=cfg["training"]
        )
        trainer.test(lit_model, dataloaders=test_loader)


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
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers for the Dataloader",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="If a model path is provided, this will execute the test run on said model.",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        type=str,
        default="config/cfg.yml",
        help="Path to model config.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
