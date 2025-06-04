import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image as ImageType
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

import holo.infra.util.paths as paths
from holo.infra.util.types import Q_, AnalysisType, DisplayType, UserDevice

logger = logging.getLogger(__name__)

HOLO_DEF = paths.MW_data()


@dataclass
class AutoConfig:
    """Class for storing torch options in the autofocus functions.

    Attributes:
        num_workers: How many data loading subprocesess to use in parallel.
        batch_size: How many images to process per epoch.
        val_split: Value split between prediction training and validation,
        leftover percentage is given to testing.

        crop_size: Length/width to crop the image to.
        grayscale: Is the image grayscale.
        opt_lr: The optimizers defined learning rate.
        opt_weight_decay: The optimizers defined learning rate.
        sch_factor: Amount to reduce the learning rate upon a plateau of improvement.
        sch_patience: Number of epochs considered a plateau for the scheduler to
        reduce the learning rate.

        num_classes: Number of classifications that can be made (only one for regression).
        backbone: The model itself.
        auto_method: Classification or regression training for the model.
        out_dir: String of path where the trained models ought to be saved.

    """

    analysis: AnalysisType = AnalysisType.CLASS
    backbone: str = "efficientnet_b4"
    batch_size: int = 16
    crop_size: int = 224
    device_user: UserDevice = UserDevice.CUDA
    epoch_count: int = 10
    grayscale: bool = True
    meta_csv_strpath: str = (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix()
    num_classes: int = 1
    num_workers: int = 2
    opt_lr: float = 5e-5
    opt_weight_decay: float = 1e-2
    out_dir: str = "checkpoints"
    sch_factor: float = 0.1
    sch_patience: int = 5
    val_split: float = 0.2
    fixed_seed: bool = True

    # Use default_factory for mutable fields and populate in __post_init__
    data: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate dictionary fields after initialization."""
        self.data.update(
            {
                "analysis": self.analysis,
                "backbone": self.backbone,
                "batch_size": self.batch_size,
                "crop_size": self.crop_size,
                "grayscale": self.grayscale,
                "meta_csv_strpath": self.meta_csv_strpath,
                "num_workers": self.num_workers,
                "num_classes": self.num_classes,
                "sch_factor": self.sch_factor,
                "sch_patience": self.sch_patience,
                "val_split": self.val_split,
            }
        )
        self.optimizer.update(
            {
                "opt_lr": self.opt_lr,
                "opt_weight_decay": self.opt_weight_decay,
                "sch_factor": self.sch_factor,
                "sch_patience": self.sch_patience,
            }
        )

    def device(self) -> Literal["cuda", "cpu"]:
        """Return the device to use in autofocus training."""
        actual_device = "cpu"  # Default to CPU
        if self.device_user == UserDevice.CUDA:
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                logger.warning("CUDA specified but not available, using CPU instead.")
        logger.info(f"Using device: {actual_device}")
        return actual_device


@dataclass
class CoreTrainer:
    """Class to hold specifically all the training information."""

    evaluation_metric: npt.NDArray[np.float64] | float
    model: Module
    loss_fn: Any
    optimizer: Optimizer
    scheduler: Any
    train_ds: Dataset[tuple[ImageType, np.float64]]
    train_loader: DataLoader[tuple[ImageType, np.float64]]
    val_ds: Dataset[tuple[ImageType, np.float64]]
    val_loader: DataLoader[tuple[ImageType, np.float64]]
    z_sig: Q_
    z_mu: Q_


@dataclass
class PlotPred:
    """Class for storing plotting information."""

    z_test_pred: list[np.float64]
    z_test: list[np.float64]
    z_train_pred: list[np.float64]
    z_train: list[np.float64]
    zerr_train: list[np.float64]
    zerr_test: list[np.float64]
    bin_edges: npt.NDArray[np.float64] | None
    title: str
    path_to_plot: str
    display: DisplayType | str


@dataclass
class GatherZ:
    """Class for storing z values retrieved from gather_z_values."""

    model: Module
    analysis: AnalysisType
    t_loader: DataLoader[tuple[ImageType, np.float64]]
    v_loader: DataLoader[tuple[ImageType, np.float64]]
    usr_device: Literal["cuda", "cpu"]
    bin_centers_phys: npt.NDArray[np.float64] | None = None
    z_mu_phys: float | None = None
    z_sig_phys: float | None = None


def load_obj(
    json_name: str = "train_data", dataclass: str = "PlotPred"
) -> list[PlotPred | GatherZ]:
    """Load saved json containing object of class plotPred or GatherZ."""
    try:
        with open(json_name) as file:
            if dataclass == "PlotPred":
                return [PlotPred(**x) for x in json.load(file)]
            if dataclass == "GatherZ":
                return [GatherZ(**y) for y in json.load(file)]
            raise Exception(f"Incorrect dataclass passed {dataclass}")
    except FileNotFoundError:
        logger.exception(FileNotFoundError)
        return []


def save_obj(c: PlotPred | GatherZ, json_name: str = "train_data") -> None:
    """Save object of class plotPred or GatherZ."""
    data = [asdict(x) for x in [c]]
    with open(json_name, "w") as file:
        json.dump(data, file)
