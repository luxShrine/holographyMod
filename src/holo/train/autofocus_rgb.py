import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import TypeVar

# import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image as ImageType
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import transforms

# local
import holo.util.epoch_helper as eh
from holo.analysis.metrics import gather_z_preds
from holo.data.dataset import HologramFocusDataset
from holo.util.log import logger
from holo.util.saveLoad import plotPred


def train_autofocus(
    hologram_dir: Path,
    metadata_csv: str,
    out_dir: str,
    backbone: str = "efficientnet_b4",
    crop_size: int = 512,
    val_split: float = 0.2,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cuda",
):
    """Trains an autofocus model.

    Args:
        hologram_dir: Directory containing hologram images.
        metadata_csv: Path to the metadata CSV file.
        out_dir: Directory to save checkpoints and logs.
        backbone: Model backbone name.
        crop_size: Size to crop images to.
        val_split: Fraction of data to use for validation.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate.
        num_workers: Number of data loader worker processes.
        seed: Random seed for reproducibility.
        device: Computing device ('cuda' or 'cpu').

    """
    eh.set_seed(seed)
    Path(out_dir).mkdir(exist_ok=True)  # ensure that the output directory exists, if not, create it
    path_to_checkpoint: Path = Path(out_dir) / Path("latest_checkpoint.pth")
    path_to_model: Path = Path(out_dir) / Path("best_model.pth")

    # make sure cuda is available otherwise use cpu
    if device == "cuda" and torch.cuda.is_available():
        pass
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("Failed to find device capable of using cuda, using cpu instaed")
        device = "cpu"
    logger.info(f"Using device: {device}")

    # TODO: If holograms are single‑channel, swap to transforms.Grayscale(num_output_channels=3) before normalizing
    # TODO: make a function to detect if one-channel?

    # Define Transforms
    train_transform = transforms.Compose(
        [
            # crop image down to appropriate size for transform could use RandomResizedCrop
            transforms.Resize((crop_size, crop_size)),
            # to ensure no bias for certain orientations, flip around
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # TODO: Add other relevant augmentations
            transforms.ToTensor(),  # convert to tensor
            # Assuming 3-channel conversion, each model has its own desired normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    # The evaluation tensor does not necessitate the same augmentations
    val_transform = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Prepare Dataset
    full_dataset = HologramFocusDataset(hologram_dir, metadata_csv, crop_size)
    num_classes = full_dataset.num_bins
    # Splits index into validation and train
    val_size: int = int(val_split * len(full_dataset))
    train_size: int = len(full_dataset) - val_size

    # generic type variables needed
    _T = TypeVar("_T")
    _T_co = TypeVar("_T_co", covariant=True)

    # assigns the images randomly to each subset based on intended ratio
    train_ds_untransformed, val_ds_untransformed = random_split(full_dataset, [train_size, val_size])

    # Apply transforms by creating a wrapper dataset to avoid modifying original HologramFocusDataset
    class TransformedDataset(Dataset[tuple[Any, Any]]):
        def __init__(self, subset: Subset[_T_co], transform: Callable[[Any], Any] | None):
            """Initialize the TransformedDataset.

            Args:
                subset (Subset[_T_co]): The subset of the original dataset.
                transform (Optional[Callable[[Any], Any]]): The transformation function to apply.
                It takes the first image and returns the transformed element.

            """
            self.subset: Subset[_T_co] = subset
            self.transform = transform

        def __getitem__(self, index: int) -> tuple[ImageType, int]:
            """Retrieve an item from the dataset by index and applies the transform.

            Args:
                index (int): The index of the item.

            Returns:
                Tuple[Any, Any]: A tuple containing the (potentially transformed) data
                                 and its corresponding label/target. The exact types
                                 depend on the subset and the transform.

            """
            x, y = self.subset[index]  # type: ignore
            if self.transform:
                # Transform applies to x
                x = self.transform(x)  # Type of x might change here
            return x, y  # type: ignore

        def __len__(self) -> int:
            """Return the total number of items in the dataset."""
            return len(self.subset)  # type: ignore

        def __getattr__(self, name: str):
            """Forward attribute look-ups to the underlying dataset, so items can be extracted."""
            try:
                return getattr(self.subset.dataset, name)  # type: ignore
            except AttributeError as e:
                raise AttributeError(name) from e

    # Assign the transformed datasets
    train_ds: TransformedDataset = TransformedDataset(train_ds_untransformed, train_transform)
    val_ds: TransformedDataset = TransformedDataset(val_ds_untransformed, val_transform)

    # create data loaders, which just allow the input of data to be iteratively called
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build model
    model = eh.get_model(num_classes, backbone).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()  # measures of error
    optimizer = optim.Adam(model.parameters(), lr=lr)  # common optimization algorithm, adapts learning rates
    # automatically reduces learning rate as metric slows down its improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    # setup progress monitoring
    console = Console()  # allows rich to automatically set certain items depending on terminal
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),  #  number completed
        TimeElapsedColumn(),  #  how long it has been
        eh.RateColumn(),  #   how fast batch speed
        TimeRemainingColumn(),  #  ETA
        eh.MetricColumn("batch_loss"),
        eh.MetricColumn("avg_loss", style="magenta"),
        eh.MetricColumn("val_loss", style="yellow"),
        eh.MetricColumn("val_acc", fmt="{:.2%}", style="green"),
        # MetricColumn("lr", fmt="{:.2e}", style="bright_black"),
        SpinnerColumn(),  # shows in progress tasks
        console=console,
        transient=False,  # keep the bars on screen after finishing
    )

    epoch_task = progress.add_task(
        "Epoch", total=epochs, avg_loss=0, val_loss=0, val_acc=0, lr=optimizer.param_groups[0]["lr"]
    )
    batch_task = progress.add_task("Batch", total=len(train_loader), batch_loss=0, avg_loss=0)
    # TODO: <04-27-25>
    best_acc: float = 0
    # data to save
    train_loss: list[float] = []
    val_loss: list[float] = []

    logger.info("Starting training...")

    with progress:  # allow for tracking of progress
        for epoch in range(1, epochs + 1):
            # Reset batch task at the start of each epoch
            progress.reset(batch_task, total=len(train_loader), batch_loss=0, avg_loss=0)

            # Train
            epoch_train_loss = eh.train_epoch(model, train_loader, criterion, optimizer, device, batch_task, progress)
            # validate

            # logger.info("Beginning Validation...")
            epoch_val_loss, epoch_acc = eh.validate_epoch(model, val_loader, criterion, device)

            # go to next step
            scheduler.step(epoch_acc)  # Step scheduler based on validation metric # type:ignore

            # store loss data
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

            bin_centers_tensor = torch.as_tensor(full_dataset.bin_centers)

            # Save best model
            if epoch_acc > best_acc:
                logger.info(f"Validation accuracy improved ({best_acc:.4f} -> {epoch_acc:.4f}). Saving best model...")
                time.sleep(0.5)  # give time to actually read it
                best_acc = epoch_acc
                torch.save(  # type:ignore
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "bin_centers": bin_centers_tensor,
                        "num_bins": num_classes,
                    },
                    path_to_model,
                )

            # Save latest model
            torch.save(  # type:ignore
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_bins": num_classes,
                    "bin_centers": bin_centers_tensor,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "epoch_acc": epoch_acc,
                },
                path_to_checkpoint,
            )

            # update the number of completed tasks, display the quality of progress made
            progress.update(
                epoch_task,
                advance=1,
                avg_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                val_acc=epoch_acc,
                lr=optimizer.param_groups[0]["lr"],
            )
        # logger.info(
        #     f"Epoch {epoch}/{epochs}: Train Loss={epoch_loss_current:.4f} | \
        #         Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}"
        # )

    logger.info(f"Training complete. Best Val Acc: {best_acc:.4f}")
    loss_hist = [train_loss, val_loss]  # return the history

    train_z_pred, train_z_true = gather_z_preds(model, train_loader, train_ds, device)  # type: ignore
    val_z_pred, val_z_true = gather_z_preds(model, val_loader, val_ds, device)  # type: ignore

    # WARN: potentially incorrect typing
    val_z_pred_list = val_z_pred.tolist()
    train_z_pred_list = train_z_pred.tolist()
    val_z_true_list = val_z_true.tolist()
    train_z_true_list = train_z_true.tolist()

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
        str(Path(out_dir) / Path("focus_depth_actual_vs_pred.png")),
        True,
    )

    return model, loss_hist, plot_data_obj
