"""Training module can handle training and evaluating on multiple gpus and plotting output function."""

import argparse
import operator
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset

from icfelab.dataset import SampleDataset
from icfelab.model import FunctionEstimator, Normalizer
from icfelab.trainer import TransformerTrainer, collate_fn
from icfelab.utils import plot_single_prediction, create_covariance, plot_full_data
from icfelab.utils import run_processes, load_cfg, initialize_random_split, load_lzma_json_data


def run_multiple_gpus(args: argparse.Namespace) -> None:
    """Launch a process for each gpu."""
    processes = [Process(target=train, args=(args, i)) for i in range(args.gpus)]
    run_processes(
        processes
    )


def main() -> None:
    """Launch processes for each gpu if possible, otherwise call train directly."""
    args = get_args()

    if args.log_to_file:
        setup_logging(args)

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


def setup_logging(args):
    log_path = Path(f"logs/cmd/{args.name}")
    log_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    out_path = log_path / f"{start_time}.out"
    err_path = log_path / f"{start_time}.err"
    sys.stdout = open(out_path, "w", encoding="utf-8")
    sys.stdout = open(err_path, "w", encoding="utf-8")
    if Path("latest.out").exists():
        os.remove("latest.out")
    if Path("latest.err").exists():
        os.remove("latest.err")
    os.symlink(out_path, "latest.out")
    os.symlink(err_path, "latest.err")


def train(args: argparse.Namespace, device_id: Union[int, str]) -> None:
    """Initialize config, datasets and dataloader and run the lightning trainer."""
    cfg, ckpt_dir, eval_batch_size, lit_model, model, test_dataset, test_loader, train_loader, val_loader = (
        init_training(args))

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename=f"{device_id}-{{epoch}}",
    )

    trainer = get_trainer(args, checkpoint_callback, device_id)

    if args.eval:
        evaluate(args, cfg, eval_batch_size, test_dataset,
                 test_loader, trainer)
    else:
        # pylint: disable=possibly-used-before-assignment
        trainer.fit(
            model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        eval_train(args, cfg, checkpoint_callback, ckpt_dir, eval_batch_size, model, test_dataset,
                   test_loader, trainer)


def get_trainer(args: argparse.Namespace, checkpoint_callback: ModelCheckpoint, device_id: Union[int, str]) -> Trainer:
    """
    Get lightning trainer.
    """
    logger = TensorBoardLogger(f"logs/{args.name}", name=f"{device_id}")
    if device_id != "cpu":
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            accelerator="gpu",
            devices=[device_id],  # type: ignore
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
    return trainer


def eval_train(args: argparse.Namespace, cfg: dict, checkpoint_callback: ModelCheckpoint, ckpt_dir: Path,
               eval_batch_size: int, model: FunctionEstimator, test_dataset: Dataset,
               test_loader: DataLoader,
               trainer: Trainer) -> None:
    """
    Evaluate model on test dataset and plot a few predicted functions.
    """
    cfg["eval"]["model_path"] = Path(checkpoint_callback.best_model_path).name
    with open(ckpt_dir / "model.yml", "w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    # pylint: disable=no-value-for-parameter
    lit_model = TransformerTrainer.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        model=model,
        hyper_parameters=cfg["training"]
    )
    print()
    trainer.test(lit_model, dataloaders=test_loader)

    prepare_predictions(args, cfg, eval_batch_size, lit_model, test_dataset, test_loader, trainer)


def prepare_predictions(args, cfg, eval_batch_size, lit_model, test_dataset, test_loader, trainer):
    gp_results = []
    rbf_scales = []
    loss_values = []
    for function in test_loader.dataset.data:
        # todo: normalize this as well?
        data = function["input"]
        rbf_scales.append(function["rbf_scale"])
        # rbf_kernel = RBF(length_scale=function["rbf_scale"])
        # gp = GaussianProcessRegressor(kernel=rbf_kernel, alpha=0.1)
        # gp.fit(np.array(data["indices"])[:, None], np.array(data["values"])[:, None])
        # gp_results.append(gp.predict(np.arange(cfg["grid_length"])[:, None], return_std=True))
        gp_results.append((data["indices"], data["values"]))
    #     loss_values.append(torch.sqrt(mse_loss(torch.tensor(gp_results[-1][0]), torch.tensor(function["target"])))) # todo do this all properly
    # print("GP RMSE", torch.tensor(loss_values).mean())
    print(torch.tensor(rbf_scales, dtype=torch.float).mean())
    predict_dataset = SampleDataset(test_dataset.data[:12])
    predict_gp_results = gp_results[:12]
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=1,
        persistent_workers=True,
    )
    plot_predictions(args, lit_model, predict_loader, trainer, predict_gp_results, rbf_scales)


def plot_predictions(args: argparse.Namespace, lit_model: TransformerTrainer, predict_loader: DataLoader,
                     trainer: Trainer, gp_results: List[Tuple[np.ndarray, np.ndarray]], rbf_scales: List[float]) -> None:
    """
    Unpack predicted batches and plot predicted functions.
    """
    # todo: remove padding points
    predictions = trainer.predict(lit_model, predict_loader)
    number = 0
    pred_plot_path = Path(f"data/{args.name}")
    pred_plot_path.mkdir(parents=True, exist_ok=True)
    for batch in predictions:  # type: ignore
        prediction = torch.squeeze(batch[0])
        indices, values, target, _ = batch[1]
        std = torch.squeeze(batch[2])
        indices = torch.squeeze(indices)
        values = torch.squeeze(values)
        target = torch.squeeze(target)

        for i, pred_data in enumerate(prediction):
            plot_full_data(pred_data, std[i], target[i], indices[i], values[i],  # type: ignore
                           pred_plot_path / f"{number}", gp_results[i][0], gp_results[i][1], rbf_scales[i])
            number += 1


def evaluate(args: argparse.Namespace, cfg: dict, eval_batch_size: int, test_dataset: Dataset, test_loader: DataLoader,
             trainer: Trainer) -> None:
    """Evaluate model on test dataset and plot a few predicted functions. This function loads the model directly from
    disk."""
    model_path = Path(args.eval) / cfg["eval"]["model_path"]
    model = FunctionEstimator(cfg["encoder"]["dim"], cfg["encoder"]["num_head"], cfg["encoder"]["num_layers"],
                              cfg["encoder"]["dim_feedforward"], gaussian=args.gaussian).train()
    # pylint: disable=no-value-for-parameter
    lit_model = TransformerTrainer.load_from_checkpoint(
        checkpoint_path=model_path, model=model, hyper_parameters=cfg["training"]
    )
    trainer.test(lit_model, dataloaders=test_loader)
    prepare_predictions(args, cfg, eval_batch_size, lit_model, test_dataset, test_loader, trainer)


def init_training(args: argparse.Namespace) -> tuple:
    """Initialize training by loading config and setting up the model and Dataloaders"""
    torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    if not args.eval:
        cfg = load_cfg(config_path)
        full_eval = False
    else:
        full_eval = args.full_eval
        cfg = load_cfg(Path(args.eval) / "model.yml")
    ckpt_dir = Path(f"models/{args.name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_dataset, train_dataset, validation_dataset = get_datasets(cfg, data_path, full_eval)
    model = FunctionEstimator(cfg["encoder"]["dim"], cfg["encoder"]["num_head"], cfg["encoder"]["num_layers"],
                              cfg["encoder"]["dim_feedforward"], gaussian=args.gaussian).train()

    batch_size = cfg["training"]["batch_size"]
    eval_batch_size = cfg["eval"]["batch_size"]
    lit_model = TransformerTrainer(model, cfg["training"])
    train_loader, val_loader = None, None
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
    return cfg, ckpt_dir, eval_batch_size, lit_model, model, test_dataset, test_loader, train_loader, val_loader


def get_datasets(cfg: dict, data_path: Path, full_eval: bool) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """
    Split data into train and validation datasets
    """
    data = load_lzma_json_data(data_path)
    indices, splits = initialize_random_split(len(data), cfg["training"]["dataset_ratio"])
    getter = operator.itemgetter(*indices[: splits[0]])
    train_dataset = SampleDataset(getter(data))
    getter = operator.itemgetter(*indices[splits[0]: splits[1]])
    validation_dataset = SampleDataset(getter(data))
    getter = operator.itemgetter(*indices[splits[1]:])
    test_dataset = SampleDataset(getter(data))

    if full_eval:
        test_dataset = SampleDataset(data)
    return test_dataset, train_dataset, validation_dataset


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
        help="If a model path is provided, this will execute the test run on said model, using the provided config file.",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        type=str,
        default="config/cfg.yml",
        help="Path to model config. When eval is active, this will be ignored.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="If true, use gaussian nll loss for training.",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="If true, use entire dataset for evaluation. Only active in eval mode.",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="If true, creates own err and out files for logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.random.manual_seed(42)
    main()
