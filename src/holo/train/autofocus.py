import os
import pathlib as Path
import random

# Imports needed for type hinting
from collections.abc import Callable
from typing import Any
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image as ImageType
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import SpinnerColumn
from rich.progress import Task
from rich.progress import TaskID
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.text import Text
from timm import create_model
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import models
from torchvision import transforms

# local
from holo.analysis.metrics import plot_actual_versus_predicted
from holo.data.dataset import HologramFocusDataset
from holo.util.log import logger

# TODO: move the progress and/or helper functions to own file?


class RateColumn(ProgressColumn):
    """Custom class for creating rate column."""

    def render(self, task: Task) -> Text:
        """Render the speed of batch processing."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.2f} batch/s", style="progress.data")


class MetricColumn(ProgressColumn):
    """Render any numeric field kept in task.fields (e.g. 'loss', 'acc', 'lr')."""

    def __init__(self, name: str, fmt: str = "{:.4f}", style: str = "cyan"):
        super().__init__()
        self.name, self.fmt, self.style = name, fmt, style

    def render(self, task: Task):
        val = task.fields.get(self.name)
        if val is None:
            return Text("–")
        return Text(self.fmt.format(val), style=self.style)


def gather_z_preds(
    model: Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    dataset: HologramFocusDataset,
    device: str,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Combine model predictions info format appropriate for comparison.

    Args:
        model (Module): Class of neural network used for prediction.
        loader (DataLoader): Iterable that contains dataset samples.
        dataset (HologramFocusDataset): Custom dataset object for passing in bin values.
        device (str): Device used for analysis.

    Returns:
        tuple[NDArray, NDArray]: The z-value predictions and truth values.

    """
    model.eval()  # set model into evaluation rather than training mode
    z_preds, z_true = [], []

    with torch.no_grad():
        for x, y in loader:  # y is class index
            # non_blocking corresponds to allowing for multiple tensors to be sent to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)  # pass in data to model

            # pick the form that matches network
            if out.ndim == 2:
                # if model outputs 2D tensor of scores/probabilities, set to 1D
                cls_pred = out.argmax(dim=1).cpu()
            else:
                # if model outputs tensor already including an index, "squeeze" out anything but the data
                cls_pred = out.squeeze().cpu().long()

            # convert z to um from object parameters
            z_pred = dataset.bin_centers[cls_pred.numpy()] * 1000
            z_tgt = dataset.bin_centers[y.cpu().numpy()] * 1000

            # store each of these values
            z_preds.append(z_pred)
            z_true.append(z_tgt)

    return np.concatenate(z_preds), np.concatenate(z_true)


def set_seed(seed: int = 42):
    """Assign a random seed to all three backends, torch, numpy and pythons native random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(num_classes: int, backbone: str = "efficientnet_b4"):
    """Load a pre-trained model and adapts its final layer for the given number of classes.

    Args:
        num_classes: The number of output classes.
        backbone: The name of the model backbone to use (e.g., 'efficientnet_b4', 'resnet50', 'vit_base_patch16_224').

    Returns:
        The PyTorch model, with the mapping of the transformation assigned.

    """
    # if type is EfficientNet | resnet | vit, load respective pre-trained model
    if backbone.startswith("efficientnet"):
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features  # size of each input sample
        # Create the linear layer (via affine linear transformation), num_classes is the size of each output sample
        fc = nn.Linear(in_feats, num_classes)
        # fc refers to "fully connected layer" the layer that performs the mapping of inputs to outputs
        model.classifier[1] = fc
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone.startswith("vit"):
        model = create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def train_epoch(
    model: Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: Module,
    optimizer: Optimizer,
    device: str,
    task_id: TaskID,
    prog: Progress,
) -> float:
    """Trains model over one epoch.

    Args:
        model (Module): The model which ought to undergo training.
        loader (DataLoader): Iterable that contains the portion of training dataset samples.
        criterion (Module): Measurer of error, as prediction probability diverges, this will generate greater values.
        optimizer (Optimizer): Algorithm used to steer the model in the correct direction.
        device (str): Device used for training (cuda or CPU).
        task_id (TaskID): Identifier for what task to update on with rich's progress bar.
        prog (Progress): Which progress object to update.

    Returns:
        float: Average loss of the model after epoch completes.

    """
    model.train()
    loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)  # associate torch tensor with device

        optimizer.zero_grad()  # reset gradients each loop
        outputs = model(imgs)  # pass in tensors of images, to get output of tensor data
        loss_current = criterion(outputs, labels)  # find loss
        loss_current.backward()  # compute the gradient of the loss
        optimizer.step()  # compute one step of the optimization algorithim

        loss += loss_current.item() * imgs.size(0)  # sum the value of the loss, scaled to the size of image tensor
        prog.update(task_id, advance=1, loss=f"{loss_current.item():.4f}")  # update progress bar

    # Check if loader.dataset exists and is not None before calculating length
    dataset_len = len(loader.dataset) if loader.dataset is not None else 0
    if dataset_len == 0:
        return 0.0  # Return zero loss and accuracy if dataset is empty

    avg_loss: float = loss / float(dataset_len)  # otherwise calcuate avg loss normally
    return avg_loss


# TODO: <04-27-25>
def validate_epoch(
    model: Module, loader: DataLoader[tuple[Tensor, Tensor]], criterion: Module, device: str
) -> tuple[float, float]:
    """Validate and compute the accuracy of one epoch."""
    model.eval()
    loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():  # reduces memory consumption when doing inference, as is the case for validation
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            current_loss = criterion(output, labels)
            loss += current_loss.item() * imgs.size(0)

            preds = output.argmax(dim=1)
            # Ensure comparison is appropriate if labels are not single integers per sample
            # WARN: If labels have a different shape, total calculation might need adjustment
            correct += (preds == labels).sum().item()
            total += labels.numel()  # Use numel() for total elements

    dataset_len = len(loader.dataset) if loader.dataset is not None else 0
    if dataset_len == 0:
        return 0.0, 0.0  # Return zero loss and accuracy if dataset is empty

    avg_loss = loss / float(dataset_len)
    acc = correct / total if total > 0 else 0.0  # Ensure total is not zero before division
    return avg_loss, acc


def train_autofocus(
    hologram_dir: Path.Path,
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
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)  # ensure that the output directory exists, if not, create it

    # make sure cuda is available otherwise use cpu
    if device == "cuda" and torch.cuda.is_available():
        pass
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("Failed to find device capable of using cuda, using cpu instaed")
        device = "cpu"
    logger.info(f"Using device: {device}")

    # TODO: <04-27-25>
    # Define Transforms
    # Note: Adjust normalization mean/std if holograms are single-channel or have different stats
    train_transform = transforms.Compose(
        [
            # crop image down to appropriate size for transform could use RandomResizedCrop
            transforms.Resize((crop_size, crop_size)),
            # to ensure no bias for certain orientations, flip around
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # TODO: Add other relevant augmentations if needed
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Assuming 3-channel conversion, each model has its own desired normalization
            # TODO: make a function to detect if black and white thus needing one-channel
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

    # TODO: <04-27-25>?
    # Prepare Datasets + Data Loaders
    full_dataset = HologramFocusDataset(
        hologram_dir, metadata_csv, crop_size
    )  # Pass crop_size if needed by Dataset internals, transforms handle final size
    num_classes = full_dataset.num_bins  # Assuming dataset calculates this
    val_size: int = int(val_split * len(full_dataset))
    train_size: int = len(full_dataset) - val_size

    # TODO: <04-27-25>?
    _T = TypeVar("_T")
    _T_co = TypeVar("_T_co", covariant=True)

    train_ds_untransformed, val_ds_untransformed = random_split(full_dataset, [train_size, val_size])

    # TODO: <04-27-25>?
    # Apply transforms - Create wrapper datasets to avoid modifying HologramFocusDataset to accept transform
    class TransformedDataset(Dataset[tuple[Any, Any]]):  # Use the TypeVar for the return type
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
            # Type hints for x, y depend on what subset.__getitem__ returns
            x, y = self.subset[index]
            if self.transform:
                # TODO:
                # Assuming transform applies to x
                x = self.transform(x)  # Type of x might change here
            # The return type depends on the type of y and the type of x after transformation
            return x, y

        def __len__(self) -> int:
            """Return the total number of items in the dataset."""
            return len(self.subset)

        def __getattr__(self, name):
            """Forward attribute look-ups we don't have to the underlying dataset, so items can be extracted."""
            try:
                return getattr(self.subset.dataset, name)
            except AttributeError as e:  # keep the default behaviour
                raise AttributeError(name) from e

    # Assign the transformed datasets
    train_ds: TransformedDataset = TransformedDataset(train_ds_untransformed, train_transform)
    val_ds: TransformedDataset = TransformedDataset(val_ds_untransformed, val_transform)

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
    model = get_model(num_classes, backbone).to(device)

    # TODO: <04-27-25>
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # common optimization algorithm, adapts learning rates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    # setup progress monitoring
    console = Console()  # allows rich to automatically set certain items depending on terminal
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),  #  number completed
        TimeElapsedColumn(),  #  how long it has been
        RateColumn(),  #   how fast batch speed
        TimeRemainingColumn(),  #  ETA
        MetricColumn("batch_loss"),
        MetricColumn("avg_loss", style="magenta"),
        MetricColumn("val_loss", style="yellow"),
        MetricColumn("val_acc", fmt="{:.2%}", style="green"),
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
            epoch_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, batch_task, progress)
            # validate
            epoch_val_loss, epoch_acc = validate_epoch(model, val_loader, criterion, device)

            # go to next step
            scheduler.step(epoch_acc)  # Step scheduler based on validation metric

            # store loss data
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

            # Save best model
            if epoch_acc > best_acc:
                logger.info(f"Validation accuracy improved ({best_acc:.4f} -> {epoch_acc:.4f}). Saving best model...")
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

            # Save latest model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "epoch_acc": epoch_acc,
                },
                os.path.join(out_dir, "latest_checkpoint.pth"),
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
    hist = [train_loss, val_loss]  # return the history

    # max_px = 255.0

    train_z_pred, train_z_true = gather_z_preds(model, train_loader, train_ds, device)
    val_z_pred, val_z_true = gather_z_preds(model, val_loader, val_ds, device)

    # WARN:
    print("unique train preds:", np.unique(train_z_pred)[:10], "…")
    print("unique val   preds:", np.unique(val_z_pred)[:10], "…")
    print("unique train true:", np.unique(train_z_true)[:10], "…")

    # train_nrmsd, train_psnr = error_metric(train_tgts, train_preds, max_px)
    # val_nrmsd, val_psnr = error_metric(val_tgts, val_preds, max_px)
    # print(f"[metrics]  Train  NRMSD={train_nrmsd:.4f} | PSNR={train_psnr:.2f} dB")
    # print(f"[metrics]  Val    NRMSD={val_nrmsd:.4f} | PSNR={val_psnr:.2f} dB")

    # Plot Actual vs Predicted and save to output dir
    plot_actual_versus_predicted(
        y_test_pred=val_z_pred,
        y_test=val_z_true,
        y_train_pred=train_z_pred,
        y_train=train_z_true,
        title="Actual vs Predicted Focus (µm)",
        save_fig=True,
        fname=os.path.join(out_dir, "focus_depth_actual_vs_pred.png"),
    )
    return model, hist
