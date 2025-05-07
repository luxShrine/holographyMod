import logging
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
from holo.util.list_helper import print_list
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
        full_dataset = HQDLHMDataset(meta_csv_name, crop_size)
    else:
        logging.exception(
            "Dataset's other than",
            print_list(known_dataset_forms),
            "might not be implemented properly, try formatting dataset as previously mentioned datasets, and \
            try again with one of their opitons.",
        )
        raise Exception
    # extra check to see if dataset is loaded properly, doesn't contain surprises
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(np.unique(full_dataset.bin_ids).size)
        logger.debug(full_dataset.bin_edges[:10])

    # get centers of dataset for parsing
    data_bin_centers = full_dataset.bin_centers

    # Splits index into validation and train
    dataset_size: int = len(full_dataset)
    val_size: int = int(val_split * dataset_size)
    train_size: int = dataset_size - val_size
    test_size: int = dataset_size - (train_size + val_size)

    # assigns the images randomly to each subset based on intended ratio
    train_ds_untransformed, val_ds_untransformed, test_ds_untransformed = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    val_transform, train_transform = _get_transformation(crop_size, grayscale)

    # Assign the transformed datasets
    train_ds: TransformedDataset = TransformedDataset(train_ds_untransformed, train_transform)
    val_ds: TransformedDataset = TransformedDataset(val_ds_untransformed, val_transform)
    test_ds: TransformedDataset = TransformedDataset(test_ds_untransformed, val_transform)

    # create data loaders, which just allow the input of data to be iteratively called
    selected_train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    selected_val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    selected_test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return selected_train_loader, selected_val_loader, selected_test_loader, data_bin_centers, train_ds, val_ds


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
        raise NameError(
            logger.exception("Selected optimizer not currently implemeted, try with: ", print_list(known_optimizers))
        )
    else:
        # common optimization algorithm, adapts learning rates
        selected_optimizer = torch.optim.AdamW(model.parameters(), lr=opt_lr, weight_decay=opt_weight_decay)
        # automatically reduces learning rate as metric slows down its improvement
        selected_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            selected_optimizer, mode="max", factor=sch_factor, patience=sch_patience
        )
        return selected_optimizer, selected_scheduler


def _create_loss_function(model_goal: str = "reg"):
    """Create manager of loss.

    Args:
        model_goal: "reg" or "class" The type of analysis the model is doing (Classification or Regression).

    Returns:
        The criterion for loss measurements.

    """
    if model_goal != "reg":
        return nn.CrossEntropyLoss()
    else:
        return nn.SmoothL1Loss()


def train_autofocus_refactored(config: AutoConfig):
    """Properly call desired autofocus training configuration."""
    #
    seed: int = 42
    eh.set_seed(seed)

    # Setup data
    train_loader, val_loader, test_loader, bin_centers, train_ds, val_ds = _setup_dataloaders(**config.data)

    # check to see if testing is expected
    try:
        if len(test_loader) > 0:
            logger.warning(f"test loader contains {len(test_loader)} items")
        else:
            pass
    except Exception as e:
        logger.exception(e)

    # setup paths for model checkkpoints
    Path(config.out_dir).mkdir(exist_ok=True)  # ensure that the output directory exists, if not, create it
    path_to_checkpoint: Path = Path(config.out_dir) / Path("latest_checkpoint.pth")
    path_to_model: Path = Path(config.out_dir) / Path("best_model.pth")

    # Setup model, optimizer, loss function (details depend on your setup)
    model = eh.get_model(**config.model)
    optimizer, scheduler = _create_optimizer(model, **config.optimizer)
    loss_fn = _create_loss_function(config.auto_method)  # measures of error

    # Setup progress display
    progress_bar = eh.setup_rich_progress(config.auto_method)

    # WARN: move prog to epoch? <luxShrine>######################################################################
    # epoch_task = progress.add_task(
    #     "Epoch", total=epochs, avg_loss=0, val_loss=0, val_acc=0, lr=optimizer.param_groups[0]["lr"]
    # )
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
    #########################################################################################################

    # TODO: Define autofocus-specific callbacks/evaluators
    # autofocus_evaluator = AutofocusMetrics(gather_z_fn=gather_z_preds, plot_fn=plotPred)

    # Run training
    # setup progress monitoring
    val_metrics = [0, 0]
    best_val_metric = 0
    train_loss: list[float] = []
    val_loss: list[float] = []
    logger.info("Starting training...")
    with progress_bar:  # allow for tracking of progress
        for epoch in range(config.epoch_count):
            # Reset batch task at the start of each epoch
            progress_bar.reset(batch_task, total=len(train_loader), batch_loss=0, avg_loss=0)
            # Train
            epoch_train_loss = eh.train_epoch(
                model, train_loader, loss_fn, optimizer, config.device(), batch_task, progress_bar
            )
            # validate, val_metrics[0] = avg_loss; val_metrics[1] = mae or accuracy
            val_metrics = eh.validate_epoch(model, val_loader, loss_fn, config.device())
            epoch_val_loss = val_metrics[0]  # this is always the value across different types of training

            # go to next step
            scheduler.step(val_metrics[1])  # Step scheduler based on validation metric # type:ignore

            # store loss data
            train_loss.append(epoch_train_loss)
            val_loss.append(val_metrics[0])

            bin_centers_tensor = torch.as_tensor(bin_centers)

            logger.debug(f"Val MAE/Acc : {val_metrics[1] * 1e6:.2f} µm")  # convert metres → µm

            # Save best model
            if val_metrics[1] < best_val_metric:
                best_val_metric = val_metrics[1]
                torch.save(  # type:ignore
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "bin_centers": bin_centers_tensor,
                        "num_bins": config.num_classes,
                    },
                    path_to_model,
                )

            # Save latest model
            torch.save(  # type:ignore
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_bins": config.num_classes,
                    "bin_centers": bin_centers_tensor,
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
                    f"Epoch {epoch}/{config.epoch_count}: Train Loss={epoch_train_loss:.4f} | \
                    Val Loss={epoch_val_loss:.4f} | Val Mae/Acc={val_metrics[1]:.4f}"
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

    logger.info(f"Training complete. Best Validation Metric: {best_val_metric:.4f}")
    loss_hist = [train_loss, val_loss]  # return the history

    train_z_pred, train_z_true = gather_z_preds(model, train_loader, train_ds, config.device)  # type: ignore
    val_z_pred, val_z_true = gather_z_preds(model, val_loader, val_ds, config.device)  # type: ignore

    # force type
    val_z_pred_list = convert_array_tolist_type(val_z_pred, float)
    train_z_pred_list = convert_array_tolist_type(train_z_pred, float)
    val_z_true_list = convert_array_tolist_type(val_z_true, float)
    train_z_true_list = convert_array_tolist_type(train_z_true, float)

    # WARN: debug
    print("unique train preds:", np.unique(train_z_pred)[:10], "…")
    print("unique val   preds:", np.unique(val_z_pred)[:10], "…")
    print("unique train true:", np.unique(train_z_true)[:10], "…")

    # Plot Actual vs Predicted and save to output dir
    train_err = np.abs(train_z_pred - train_z_true)
    val_err = np.abs(val_z_pred - val_z_true)

    # save training data for plotting
    plot_data_obj = plotPred(
        val_z_pred_list,
        train_z_pred_list,
        val_z_true_list,
        train_z_true_list,
        train_err.tolist(),
        val_err.tolist(),
        "Actual vs Predicted Focus (µm)",
        str(Path(config.out_dir) / Path("focus_depth_actual_vs_pred.png")),
        True,
    )

    return model, loss_hist, plot_data_obj
