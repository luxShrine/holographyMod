from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch

from holo.util.log import logger


@dataclass
class AutoConfig:
    """Class for storing torch options in the autofocus functions.

    Attributes:
        num_workers: How many data loading subprocesess to use in parallel.
        batch_size: How many images to process per epoch.
        val_split: Value split between prediction training and validation, leftover percentage is given to testing.
        dataset_name: Name of the dataset, corresponds to schema of the structure.
        crop_size: Length/width to crop the image to.
        grayscale: Is the image grayscale.
        optimizer_type: The name of the optimaizer to use.
        opt_lr: The optimizers defined learning rate.
        opt_weight_decay: The optimizers defined learning rate.
        sch_factor: Amount to reduce the learning rate upon a plateau of improvement.
        sch_patience: Number of epochs considered a plateau for the scheduler to reduce the learning rate.
        num_classes: Number of classifications that can be made (only one for regression).
        backbone: The model itself.
        auto_method: Classification or regression training for the model.
        out_dir: String of path where the trained models ought to be saved.

    """

    auto_method: str = "reg"
    backbone: str = "efficientnet_b4"
    batch_size: int = 16
    crop_size: int = 224
    dataset_name: str = "hqdlhm"
    device_user: str = "cuda"
    epoch_count: int = 10
    grayscale: bool = True
    meta_csv_name: str = "ODP-DLHM-Database.csv"
    num_classes: int = 1
    num_workers: int = 4
    opt_lr: float = 5e-5
    opt_weight_decay: float = 1e-2
    optimizer_type: str = "adam"
    out_dir: str = "checkpoints"
    sch_factor: float = 0.1
    sch_patience: int = 5
    val_split: float = 0.2

    # Use default_factory for mutable fields and populate in __post_init__
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate dictionary fields after initialization."""
        self.data.update(
            {
                "meta_csv_name": self.meta_csv_name,
                "auto_method": self.auto_method,
                "num_workers": self.num_workers,
                "batch_size": self.batch_size,
                "val_split": self.val_split,
                "dataset_name": self.dataset_name,
                "crop_size": self.crop_size,
                "grayscale": self.grayscale,
            }
        )
        self.model.update(
            {
                "num_classes": self.num_classes,  # Note: num_workers is also in data, consider if this is intended
                "backbone": self.backbone,
            }
        )
        self.optimizer.update(
            {
                "optimizer_type": self.optimizer_type,
                "opt_lr": self.opt_lr,
                "opt_weight_decay": self.opt_weight_decay,
                "sch_factor": self.sch_factor,
                "sch_patience": self.sch_patience,
            }
        )

    def device(self):
        """Return the device to use in autofocus training."""
        actual_device = "cpu"  # Default to CPU
        if self.device_user == "cuda":
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                logger.warning("CUDA specified but not available, using CPU instead.")
        logger.info(f"Using device: {actual_device}")
        return actual_device


# TODO: class to hold specifically all the training information?
# @dataclass
# class CoreTrainer():
#     model,
#     optimizer,
#     loss_fn,
#     device=config.device,
#     progress_bar

# TODO: class to hold specifically all the metric data for plotting
# @dataclass
# class AutofocusMetrics():
#     gather_z_fn
#     plot_fn
