import random

import numpy as np
import torch
import torch.nn as nn
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskID
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from timm import create_model  # type: ignore
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models

from holo.util.list_helper import print_list
from holo.util.log import console_ as console
from holo.util.log import log_timing
from holo.util.log import logger
from holo.util.prog_helper import MetricColumn
from holo.util.prog_helper import RateColumn


def setup_rich_progress(train_type: str):
    """Setup progress monitoring for epoch."""
    if train_type == "reg":
        accuracy_measure = "val_mae"
        acc_col = MetricColumn(accuracy_measure, fmt="{:.2}", style="green")
    elif train_type == "class":
        accuracy_measure = "val_acc"
        acc_col = MetricColumn(accuracy_measure, fmt="{:.2%}", style="green")
    else:
        logger.exception(f"Failed to process training type of {train_type}, assuming type is regression.")
        accuracy_measure = "val_mae"
        acc_col = MetricColumn(accuracy_measure, fmt="{:.2}", style="green")

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),  #  number completed
        TimeElapsedColumn(),  #  how long it has been
        RateColumn(),  #   batch speed
        TimeRemainingColumn(),  #  ETA
        # eh.MetricColumn("batch_loss"),
        MetricColumn("avg_loss", style="magenta"),
        MetricColumn("val_loss", style="yellow"),
        acc_col,
        MetricColumn("lr", fmt="{:.2e}", style="bright_black"),
        SpinnerColumn(),  # shows in progress tasks
        console=console,  # allows rich to automatically set certain items depending on terminal
        transient=False,  # keep the bars on screen after finishing
    )
    return progress


def get_model(num_classes: int = 20, backbone: str = "efficientnet_b4"):
    """Load a pre-trained model and adapts its final layer for the given number of classes.

    Args:
        num_classes: The number of output classes.
        backbone: The name of the model backbone to use (e.g., 'efficientnet_b4', 'resnet50', 'vit_base_patch16_224').

    Returns:
        The PyTorch model, with the mapping of the transformation assigned.

    """
    known_backbones = ["efficientnet", "resnet50", "vit"]
    # if type is EfficientNet | resnet | vit, load respective pre-trained model
    if backbone.startswith("efficientnet"):
        selected_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = selected_model.classifier[1].in_features  # size of each input sample
        # Create the linear layer (via affine linear transformation), num_classes is the size of each output sample
        fc = nn.Linear(in_feats, num_classes)  # type: ignore
        # fc refers to "fully connected layer" the layer that performs the mapping of inputs to outputs
        selected_model.classifier[1] = fc
    elif backbone == "resnet50":
        selected_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        selected_model.fc = nn.Linear(selected_model.fc.in_features, num_classes)
    elif backbone.startswith("vit"):
        selected_model = create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    else:
        logger.exception(f"Unknown backbone: {backbone}, known backbones include: ")
        print_list(known_backbones)
        raise NameError

    return selected_model


def set_seed(seed: int = 42):
    """Assign a random seed to all three backends, torch, numpy and pythons native random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type:ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: Module,
    analysis: str,
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
        analysis (str): Type of analysis to train model for (regression "reg" or classification).
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
    torch_device = torch.device(device)
    # for n, p in model.named_parameters():
    #     if "head" not in n and "classifier" not in n:
    #         p.requires_grad = False  # first 3 epochs
    model.cuda()
    for imgs, labels in loader:
        imgs, labels = imgs.to(torch_device), labels.to(torch_device)  # associate torch tensor with device

        optimizer.zero_grad()  # reset gradients each loop
        outputs = model(imgs)  # pass in tensors of images, to get output of tensor data

        # reduce tensor size down to one dimension as we are only looking for one prediction
        if analysis == "reg":
            outputs = outputs.squeeze(1)

        loss_current = criterion(outputs, labels)  # find loss
        loss_current.backward()  # compute the gradient of the loss
        optimizer.step()  # compute one step of the optimization algorithm

        loss += loss_current.item() * imgs.size(0)  # sum the value of the loss, scaled to the size of image tensor
        prog.update(task_id, advance=1, loss=f"{loss_current.item():.4f}")  # update progress bar

    # Check if loader.dataset exists and is not None before calculating length
    dataset_len = len(loader.dataset) if loader.dataset is not None else 0
    if dataset_len == 0:
        return 0.0  # Return zero loss and accuracy if dataset is empty

    avg_loss: float = loss / float(dataset_len)  # otherwise calcuate avg loss normally
    return avg_loss


def validate_epoch(
    model: Module,
    analysis: str,
    z_sigma: float,
    z_mu: float,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: Module,
    device: str,
    # bin_centers,
) -> tuple[float, float]:
    """Validate and compute the accuracy of one epoch."""
    with log_timing("Epoch validation"):
        model.eval()
        loss: float = 0
        # correct: int = 0
        abs_err_sum: float = 0
        total: int = 0

        with torch.no_grad():  # reduces memory consumption when doing inference, as is the case for validation
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs).squeeze(1)
                current_loss = criterion(output, labels)
                loss += current_loss.item() * imgs.size(0)

                # de-normalise as tensors
                z_pred_m = output * z_sigma + z_mu  # still on GPU
                z_true_m = labels * z_sigma + z_mu

                abs_err_sum += torch.sum(torch.abs(z_pred_m - z_true_m)).item()
                total += z_true_m.numel()  # or imgs.size(0)

    dataset_len = len(loader.dataset) if loader.dataset is not None else 0
    if dataset_len == 0:
        return 0.0, 0.0  # Return zero loss and accuracy if dataset is empty

    avg_loss = loss / total
    mae = abs_err_sum / total  # in metres if z is metres
    logger.debug(f"Val MAE : {mae:.2f} m")  # convert metres → µm
    return avg_loss, mae
