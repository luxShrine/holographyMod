import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

# import numpy.typing as npt
# from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import holo.train.epoch_helper as eh

# local
from holo.analysis.metrics import gather_z_preds
from holo.data.dataset import HQDLHMDataset
from holo.data.transformed_dataset import TransformedDataset
from holo.train.auto_classes import AutoConfig
from holo.util.list_helper import convert_array_tolist_type
from holo.util.log import logger
from holo.util.saveLoad import plotPred


def _get_transformation(crop: int, gray: bool):
    # Define universal transforms
    # crop image down to appropriate size for transform could use RandomResizedCrop
    train_trans_list = [
        transforms.Resize((crop, crop)),
        # to ensure no bias for certain orientations, flip around
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # TODO: Add other relevant augmentations
        transforms.ToTensor(),  # convert to tensor
        # Assuming 3-channel conversion, each model has its own desired normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    # The evaluation tensor does not necessitate the same augmentations
    val_trans_list = [
        transforms.Resize((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if gray:
        # Define Transforms
        grayscale_trans = transforms.Grayscale(num_output_channels=3)
        # ignore type because it expects list to be of the types intiially given
        train_trans_list.insert(1, grayscale_trans)  # type: ignore
        val_trans_list.insert(1, grayscale_trans)  # type: ignore

    train_trans = transforms.Compose(train_trans_list)
    val_trans = transforms.Compose(val_trans_list)

    return train_trans, val_trans


def _setup_dataloaders(
    meta_csv_name: str,
    auto_method: str,
    num_workers: int = 4,
    batch_size: int = 16,
    val_split: float = 0.2,
    dataset_name: str = "hdqlhm",
    crop_size: int = 224,
    grayscale: bool = True,
):
    """ """

    known_dataset_forms = ["hqdlhm"]
    # Prepare Dataset
    if dataset_name in known_dataset_forms:
        full_dataset = HQDLHMDataset(meta_csv_name, crop=crop_size, mode=auto_method)
        total_ids = full_dataset.bin_ids
        z_sig = full_dataset.z_sigma
        z_mu = full_dataset.z_mu
    else:
        logger.exception(
            f"Selected dataset's may not be implemeted, try with any of the supllied items: {known_dataset_forms}"
        )
        raise Exception
    # extra check to see if dataset is loaded properly, doesn't contain surprises
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(np.unique(full_dataset.bin_ids).size)
        logger.debug(full_dataset.bin_edges_m[:10])

    if auto_method == "reg":
        # get centers of dataset for parsing
        z_val = full_dataset.z_m
        evaluation_metric = z_val
    else:
        # get centers of dataset for parsing
        data_bin_centers = full_dataset.bin_centers
        evaluation_metric = data_bin_centers

    # Splits index into validation and train
    dataset_size: int = len(full_dataset)
    val_size: int = int(val_split * dataset_size)
    train_size: int = dataset_size - val_size
    # test_size: int = dataset_size - (train_size + val_size)  # whatever is leftover (currently intended to be zero)

    train_transform, val_transform = _get_transformation(crop_size, grayscale)

    # assigns the images randomly to each subset based on intended ratio
    train_ds_untransformed, val_ds_untransformed = random_split(full_dataset, [train_size, val_size])
    # Assign the transformed datasets
    train_ds: TransformedDataset = TransformedDataset(train_ds_untransformed, train_transform, total_ids)
    val_ds: TransformedDataset = TransformedDataset(val_ds_untransformed, val_transform, total_ids)

    try:
        # create data loaders, which just allow the input of data to be iteratively called
        selected_train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        selected_val_loader = DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return selected_train_loader, selected_val_loader, evaluation_metric, train_ds, val_ds, z_sig, z_mu
    except:
        logger.exception(
            f"Error occurerd at final stage of creating dataloaders\t \
            train: {train_size} | val: {val_size} | "
        )
        raise


def _create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    opt_lr: float = 5e-5,
    opt_weight_decay: float = 1e-2,
    sch_factor: float = 0.1,
    sch_patience: int = 5,
):
    """ """
    known_optimizers = ["adam"]
    if optimizer_type not in known_optimizers:
        logger.exception(
            f"Selected optimizer not currently implemeted, try with any of the supllied items: {known_optimizers}",
        )
        raise NameError
    else:
        # common optimization algorithm, adapts learning rates
        selected_optimizer = torch.optim.AdamW(model.parameters(), lr=opt_lr, weight_decay=opt_weight_decay)
        # automatically reduces learning rate as metric slows down its improvement
        selected_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            selected_optimizer, mode="min", factor=sch_factor, patience=sch_patience
        )
        return selected_optimizer, selected_scheduler


def _create_loss_function(analysis: str = "reg"):
    """Create manager of loss.

    Args:
       analysis: "reg" or "class" The type of analysis the model is doing (Classification or Regression).

    Returns:
        The criterion for loss measurements.

    """
    if analysis != "reg":
        return nn.CrossEntropyLoss()
    else:
        return nn.SmoothL1Loss()


def train_autofocus_refactored(config: AutoConfig):
    """Properly call desired autofocus training configuration."""
    #
    seed: int = 42
    eh.set_seed(seed)
    config_as_dict = asdict(config)  # convert to dict so it can easily be passed into functions

    # Setup data
    train_loader, val_loader, eval_metric, train_ds, val_ds, z_sig, z_mu = _setup_dataloaders(**config_as_dict["data"])

    # setup paths for model checkkpoints
    Path(config.out_dir).mkdir(exist_ok=True)  # ensure that the output directory exists, if not, create it
    path_to_checkpoint: Path = Path(config.out_dir) / Path("latest_checkpoint.pth")
    path_to_model: Path = Path(config.out_dir) / Path("best_model.pth")

    # create device variable
    device = config.device()

    # Setup model, optimizer, loss function (details depend on your setup)
    model = eh.get_model(**config_as_dict["model"])
    optimizer, scheduler = _create_optimizer(model, **config_as_dict["optimizer"])
    loss_fn = _create_loss_function(config.auto_method)  # measures of error

    # Setup progress display
    progress_bar = eh.setup_rich_progress(config.auto_method)

    # TODO: move prog to epoch? <luxShrine>
    epoch_task = progress_bar.add_task(
        "Epoch",
        total=config.epoch_count,
        avg_loss=0,
        val_loss=0,
        accuracy_measure=0,
        lr=optimizer.param_groups[0]["lr"],
    )
    # batch_task = progress.add_task("Batch", total=len(train_loader), batch_loss = 0, avg_loss=0)
    batch_task = progress_bar.add_task("Batch", total=len(train_loader), avg_loss=0)

    # TODO: Define autofocus-specific callbacks/evaluators
    # autofocus_evaluator = AutofocusMetrics(gather_z_fn=gather_z_preds, plot_fn=plotPred)

    # Run training
    # setup progress monitoring
    # positive initialization for regression, negative for accuracy
    best_val_metric = float("inf")
    if config.auto_method == "reg":
        pass
    else:
        best_val_metric = -best_val_metric

    val_metrics = [0, best_val_metric]
    train_loss: list[float] = []
    val_loss: list[float] = []
    logger.info("Starting training...")
    with progress_bar:  # allow for tracking of progress
        for epoch in range(config.epoch_count):
            # Reset batch task at the start of each epoch
            progress_bar.reset(batch_task, total=len(train_loader), batch_loss=0, avg_loss=0)
            # Train
            epoch_train_loss = eh.train_epoch(
                model=model,
                analysis=config.auto_method,
                loader=train_loader,
                criterion=loss_fn,
                optimizer=optimizer,
                device=device,
                task_id=batch_task,
                prog=progress_bar,
            )
            # validate, val_metrics[0] = avg_loss; val_metrics[1] = mae or accuracy
            val_metrics = eh.validate_epoch(model, config.auto_method, z_sig, z_mu, val_loader, loss_fn, device)
            epoch_val_loss = val_metrics[0]  # this is always the value across different types of training

            # go to next step
            scheduler.step(val_metrics[1])  # Step scheduler based on validation metric # type:ignore

            # store loss data
            train_loss.append(epoch_train_loss)
            val_loss.append(val_metrics[0])

            if config.auto_method == "reg":
                # using z value
                labels_tensor = torch.as_tensor(eval_metric, dtype=torch.float32)
                logger.debug(f"Val MAE: {val_metrics[1]:.9f} µm")
            else:
                # using bin value
                labels_tensor = torch.as_tensor(eval_metric)
                logger.debug(f"Val Acc: {val_metrics[1] * 100:.2f} %")  # percent

            # Save best model
            if val_metrics[1] < best_val_metric:
                best_val_metric = val_metrics[1]
                torch.save(  # type:ignore
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "labels": labels_tensor,
                        "bin_centers": getattr(train_ds, "bin_centers", None),
                        "num_bins": config.num_classes,
                    },
                    path_to_model,
                )

            # Save latest model
            torch.save(  # type:ignore
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "bin_centers": getattr(train_ds, "bin_centers", None),
                    "num_bins": config.num_classes,
                    "labels": labels_tensor,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "val_metric": val_metrics[1],
                },
                path_to_checkpoint,
            )

            # update the number of completed tasks, display the quality of progress made
            progress_bar.update(
                epoch_task,
                advance=1,
                avg_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                val_mae=val_metrics[1],
                lr=optimizer.param_groups[0]["lr"],
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Epoch {epoch}/{config.epoch_count}: Train Loss={epoch_train_loss:.8f} | \
                    Val Loss={epoch_val_loss:.8f} | Val Mae/Acc={val_metrics[1]:.10f}"
                )

            # TODO: testing
            # Check if test.dataset exists and is not None before calculating length
            # dataset_len = len(loader.dataset) if loader.dataset is not None else 0
            # if dataset_len == 0:
            #     pass  # Return zero loss and accuracy if dataset is empty
            # else:
            #     autofocus_evaluator.evaluate(model, val_loader, epoch)

    # TODO: testing
    # if test_loader is None:
    #     pass
    # else:
    #     final_results = autofocus_evaluator.evaluate(model, test_loader, "final")
    #     logger.info(final_results)

    logger.info(f"Training complete. Best Validation Metric: {best_val_metric:.8f}")
    loss_hist = [train_loss, val_loss]  # return the history

    train_z_pred, train_z_true = gather_z_preds(model, config.auto_method, train_loader, train_ds, device)  # type: ignore
    val_z_pred, val_z_true = gather_z_preds(model, config.auto_method, val_loader, val_ds, device)  # type: ignore

    # force type
    val_z_pred_list = convert_array_tolist_type(val_z_pred, float)
    train_z_pred_list = convert_array_tolist_type(train_z_pred, float)
    val_z_true_list = convert_array_tolist_type(val_z_true, float)
    train_z_true_list = convert_array_tolist_type(train_z_true, float)

    # debug for classification bins
    if logger.isEnabledFor(logging.DEBUG) and config.auto_method == "class":
        logger.debug(
            "unique train preds:",
            np.unique(train_z_pred)[:10],
            "…unique val   preds:",
            np.unique(val_z_pred)[:10],
            "…unique train true:",
            np.unique(train_z_true)[:10],
            "…",
        )

    # Actual vs Predicted diff
    train_err = np.abs(train_z_pred - train_z_true)
    val_err = np.abs(val_z_pred - val_z_true)

    # save training data for plotting
    plot_data_obj = plotPred(
        val_z_pred_list,
        val_z_true_list,
        train_z_pred_list,
        train_z_true_list,
        train_err.tolist(),
        val_err.tolist(),
        "Actual vs Predicted Focus (µm)",
        str(Path(config.out_dir) / Path("focus_depth_actual_vs_pred.png")),
        True,
    )

    return model, loss_hist, plot_data_obj
