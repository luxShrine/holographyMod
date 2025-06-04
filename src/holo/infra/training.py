import logging
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

# from line_profiler import profile
from PIL.Image import Image as ImageType
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from timm import create_model  # Make sure timm is installed if using ViT from there
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import models
from torchvision.transforms import v2

from holo.core.metrics import gather_z_preds
from holo.infra.dataclasses import AutoConfig, CoreTrainer, GatherZ, PlotPred
from holo.infra.datamodules import HologramFocusDataset
from holo.infra.tests import test_base
from holo.infra.util.prog_helper import MetricColumn, RateColumn
from holo.infra.util.types import Q_, AnalysisType, Np1Array64, u

logger = logging.getLogger(__name__)


class TransformedDataset(Dataset[tuple[ImageType, np.float64]]):
    """Transform the input dataset by applying transfomations to it's contents.

    By wrapping the underlying dataset in this class, we can ensure that attrbutes
    retrived from this new object will undergo the expected transfomrmations while
    preserving the original data.

    Attributes:
        subset_obj: Subset[tuple[ImageType, np.float64]] Subset of the original
                    dataset.
        img_transform: v2.Compose | None  Transformation(s) to be applied to images.
        label_transform: v2.Lambda | None Transformation(s) to be applied to labels.

    """

    def __init__(
        self,
        subset_obj: Subset[tuple[ImageType, np.float64]],
        img_transform: v2.Compose | None = None,
        label_transform: v2.Lambda | None = None,
    ) -> None:
        # NOTE: v2 transformation classes != typical torch transformation classes
        self.subset_obj: Subset[tuple[ImageType, np.float64]] = subset_obj
        self.img_transform: v2.Compose | None = img_transform
        self.label_transform: v2.Lambda | None = label_transform

    @override
    def __getitem__(self, idx: int) -> tuple[Any | ImageType, Any | np.float64]:
        """Retrieve the image and label from the subest object."""
        img, label = self.subset_obj[idx]  # Gets (PIL Image, raw_label)

        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        """Get number of entries in subset."""
        return len(self.subset_obj)


def train_autofocus(a_config: AutoConfig) -> PlotPred:
    # load data
    holo_base_ds = HologramFocusDataset(
        mode=a_config.analysis,
        num_classes=a_config.num_classes,
        csv_file_strpath=a_config.meta_csv_strpath,
    )
    test_base(holo_base_ds)

    # transform that data
    # distrubute to dataloaders
    # get information to train/validate
    train_cfg: CoreTrainer = transform_ds(holo_base_ds, a_config)
    test_training_config(train_cfg, a_config)

    # For measuring evaluation: classificaton is maximizing correct bins,
    # regression is minimizing the error from expected
    best_val_metric: float = (
        float("inf") if a_config.analysis == AnalysisType.REG else -float("inf")
    )
    # train/validate, one epoch,
    avg_loss_train, avg_loss_val, metric_val, best_val_metric = train_eval_epoch(
        train_cfg, a_config, best_val_metric
    )
    logger.info(f"metric_val: {metric_val}")
    logger.info(f"avg_loss_val: {avg_loss_val}")
    logger.info(f"avg_loss_train: {avg_loss_train}")

    # regression needs std and average
    z_mu: float | None = (
        float(train_cfg.z_mu.magnitude) if a_config.analysis == AnalysisType.REG else None
    )
    z_sig: float | None = (
        float(train_cfg.z_sig.magnitude) if a_config.analysis == AnalysisType.REG else None
    )
    # For class, train_cfg.evaluation_metric is base.bin_centers
    bin_centers = holo_base_ds.bin_centers if a_config.analysis == AnalysisType.CLASS else None

    # -- Get & Save Training Data for Plotting ---------------------------------------------------
    # store for ease of use during debugging
    gather_z_obj = GatherZ(
        model=train_cfg.model,
        analysis=a_config.analysis,
        t_loader=train_cfg.train_loader,
        v_loader=train_cfg.val_loader,
        usr_device=a_config.device(),
        bin_centers_phys=bin_centers,
        z_mu_phys=z_mu,
        z_sig_phys=z_sig,
    )
    # TODO: Save the object to json/pickle/torch file to have access to
    # predictions for debugging/inspection purposes.
    return return_z(holo_base_ds, a_config, gather_z_obj)


def return_z(holo_base_ds: HologramFocusDataset, a_config: AutoConfig, gather: GatherZ):
    assert isinstance(gather, GatherZ), f"gather is not GatherZ, found {type(gather)}"
    train_z_pred, train_z_true, val_z_pred, val_z_true = gather_z_preds(**gather.__dict__)
    # TODO: get something more robust
    # Actual vs Predicted diff
    train_err: Np1Array64 = np.abs(train_z_pred - train_z_true)
    val_err: Np1Array64 = np.abs(val_z_pred - val_z_true)

    # create list from numpy arrays, cast them so typechecker knows contents
    return PlotPred(
        cast("list[np.float64]", val_z_pred.tolist()),
        cast("list[np.float64]", val_z_true.tolist()),
        cast("list[np.float64]", train_z_pred.tolist()),
        cast("list[np.float64]", train_z_true.tolist()),
        cast("list[np.float64]", train_err.tolist()),
        cast("list[np.float64]", val_err.tolist()),
        cast("list[np.float64]", holo_base_ds.bin_edges.tolist()),
        "plot",
        str(Path(a_config.out_dir) / Path("plot.png")),
        "save",  # cannot serialize DisplayType to json
    )


def transform_ds(base: HologramFocusDataset, a_cfg: AutoConfig) -> CoreTrainer:
    num_labels: int = len(base)
    eval_len = int(a_cfg.val_split * num_labels)
    train_len = num_labels - eval_len
    if a_cfg.fixed_seed:
        generator = torch.Generator().manual_seed(42)
        train_indices, eval_indices = random_split(base, [train_len, eval_len], generator)
    else:
        train_indices, eval_indices = random_split(base, [train_len, eval_len])

    # TODO: evaluate implementing extra transforms <05-24-25, luxShrine>
    extra_tf: list[nn.Module] = [
        # crop random area
        v2.RandomResizedCrop(size=a_cfg.crop_size, antialias=True),
        # flip image with some probability
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # Random rotation image with some probability
        # v2.RandomRotation([1, 30]),
        # v2.RandomAdjustSharpness(sharpness_factor=2),
    ]

    common_tf: list[nn.Module] = [
        # convert PIL to tensor
        v2.PILToTensor(),
        # ToTensor preserves original datatype, this ensures it is proper input type
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(size=a_cfg.crop_size),
        # normalize across channels, expects float
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # no random
    eval_transform: v2.Compose = v2.Compose(common_tf)
    train_transform: v2.Compose = v2.Compose(extra_tf + common_tf)
    logger.debug("Train and evaluation transformations composed successfully.")

    eval_subset = Subset(base, eval_indices.indices)
    train_subset = Subset(base, train_indices.indices)
    logger.debug("Train and evaluation subset created successfully")

    # Define label transforms based on analysis type (handles normalization for REG)
    if a_cfg.analysis == AnalysisType.REG:
        # Calculate mu and sigma from the training subset's physical z-values
        # Need to access original z_m from base dataset via indices of train_subset
        train_z_physical_subset: Np1Array64 = base.z_m[train_subset.indices]
        core_z_mu_val: float = train_z_physical_subset.mean()
        core_z_sig_val: float = train_z_physical_subset.std()
        if core_z_sig_val < 1e-6:  # Avoid division by zero or very small std
            core_z_sig_val = 1.0
            logger.warning(
                "Training subset z_m standard deviation is near zero."
                + f"Setting to {core_z_sig_val} for normalization."
            )

        core_z_mu = Q_(core_z_mu_val, u.m)
        core_z_sig = Q_(core_z_sig_val, u.m)

        def _reg_label_transform_fn(z_raw_phys_val: np.float32) -> Tensor:
            """Pass in physical value, return normalized z tensor."""
            z_tensor = torch.as_tensor(z_raw_phys_val, dtype=torch.float32)
            return (z_tensor - core_z_mu.magnitude) / core_z_sig.magnitude

        # apply local function above to z values to create proper label
        final_label_transform = v2.Lambda(_reg_label_transform_fn)
        # True physical Zs for validation
        evaluation_metric = base.z_m[eval_subset.indices]
        model_output_dim = 1
    else:
        # simply convert to tensor
        final_label_transform = v2.Lambda(
            lambda z_raw_idx: torch.as_tensor(z_raw_idx, dtype=torch.long)
        )
        # Physical values of bin centers
        core_z_mu = Q_(base.z_m.mean(), u.m)
        core_z_sig = Q_(base.z_m.std(), u.m)
        evaluation_metric = base.z_bins[eval_subset.indices]
        model_output_dim = a_cfg.num_classes

    # providing the dataset with these transforms will create a new subset
    # dataset containing the transformed images
    tf_eval_ds = TransformedDataset(
        subset_obj=eval_subset,
        img_transform=eval_transform,
        label_transform=final_label_transform,
    )
    tf_train_ds = TransformedDataset(
        subset_obj=train_subset,
        img_transform=train_transform,
        label_transform=final_label_transform,
    )
    logger.debug("transformed datasets created")

    # dataset needs to be iterable in terms of pytorch, dataloader does such
    eval_dl: DataLoader[tuple[ImageType, np.float64]] = DataLoader(
        tf_eval_ds,
        batch_size=a_cfg.batch_size,
        shuffle=True,
        pin_memory=a_cfg.device == "cuda",
    )
    train_dl: DataLoader[tuple[ImageType, np.float64]] = DataLoader(
        tf_train_ds,
        batch_size=a_cfg.batch_size,
        shuffle=True,
        pin_memory=a_cfg.device == "cuda",
    )
    logger.debug("Dataloaders created successfully")
    test_loader(eval_dl)
    test_loader(train_dl)

    # loss_fn = nn.CrossEntropyLoss() if a_cfg.analysis == AnalysisType.CLASS else nn.SmoothL1Loss()
    loss_fn = nn.CrossEntropyLoss() if a_cfg.analysis == AnalysisType.CLASS else nn.SmoothL1Loss()
    model = _create_model(a_cfg.backbone, model_output_dim, a_cfg.analysis)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a_cfg.opt_lr, weight_decay=a_cfg.opt_weight_decay
    )

    # Scheduler monitors val_metrics[1]: MAE (min) for REG, Accuracy (max) for CLASS
    scheduler_mode = "min" if a_cfg.analysis == AnalysisType.REG else "max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=a_cfg.sch_factor, patience=a_cfg.sch_patience
    )

    # Create CoreTrainer
    core_trainer_config = CoreTrainer(
        evaluation_metric=evaluation_metric,  # This is Np1Array64 of z_m or bin_centers_m
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=tf_train_ds,
        train_loader=train_dl,
        val_ds=tf_eval_ds,
        val_loader=eval_dl,
        z_sig=core_z_sig,
        z_mu=core_z_mu,
    )
    logger.info(
        "[black on green]--- Epoch Variables Initialization Complete ---", extra={"markup": True}
    )
    return core_trainer_config


def test_loader(ds_loader: DataLoader[tuple[ImageType, np.float64]]) -> None:
    """Attempt to grab image and label."""
    try:
        train_features, train_labels = next(iter(ds_loader))
        logger.debug(f"Feature batch shape: {train_features.size()}")
        logger.debug(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze(1)
        img = img[1, :, :].numpy()
        label = train_labels[0]
        if label.ndim == 0:
            logger.debug(f"Sample label value: {label.item()}")
        else:
            logger.debug(f"Sample label tensor: {label}")
    except Exception as e:
        raise e


def _create_model(
    backbone_name: str, num_model_outputs: int, analysis_type: AnalysisType
) -> nn.Module:
    """Help to create and configure the model."""
    model: nn.Module
    if backbone_name.startswith("efficientnet"):
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_model_outputs)
    elif backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_model_outputs)
    elif backbone_name.startswith("vit"):
        model = create_model(backbone_name, pretrained=True, num_classes=num_model_outputs)
    else:
        raise Exception(f"Could not create model specified: {backbone_name}")
    logger.info(
        f"Model '{backbone_name}' configured with {num_model_outputs} output "
        + f"features for {analysis_type.value} analysis."
    )
    return model


def test_training_config(t_cfg: CoreTrainer, a_cfg: AutoConfig) -> None:
    # depends on analysis
    type_eval: type[npt.NDArray[np.float64] | float] = type(t_cfg.evaluation_metric)
    if a_cfg.analysis is a_cfg.analysis.CLASS:
        # must be bins, check how many, type
        assert isinstance(t_cfg.evaluation_metric, np.ndarray), (
            f"evaluation_metric is not NDArray, found {type_eval}"
        )
    else:
        # TODO: create test for regrsession <05-25-25, luxShrine>
        assert isinstance(t_cfg.evaluation_metric, float), (
            f"evaluation_metric is not float, found {type_eval}"
        )

    # TODO: create a test fort these, maybe match expected?
    # t_cfg.model
    # t_cfg.loss_fn
    # t_cfg.optimizer
    # t_cfg.scheduler

    # check type
    # TODO: check that you can pull expected items from ds
    assert isinstance(t_cfg.val_ds, TransformedDataset), (
        f"t_cfg.val_ds is not subset, found {type(t_cfg.val_ds)}"
    )
    assert isinstance(t_cfg.train_ds, TransformedDataset), (
        f"t_cfg.train_ds is not subset, found {type(t_cfg.train_ds)}"
    )

    # run through tester
    test_loader(t_cfg.train_loader)
    test_loader(t_cfg.val_loader)

    # TODO: within expected range, type
    # t_cfg.z_sig
    # t_cfg.z_mu
    # WARN: doesn't render in console correctly, interupts rich progress bar
    logger.info("[black on green]--- Training Config Validated ---", extra={"markup": True})


def train_eval_epoch(
    epoch_cfg: CoreTrainer, a_cfg: AutoConfig, best_val_metric: float
) -> tuple[float, float, float, float]:
    """Trains model over one epoch.

    Returns:
        float: Average loss of the model after epoch completes.

    """
    _ = Path(a_cfg.out_dir).mkdir(exist_ok=True)
    path_to_checkpoint: Path = Path(a_cfg.out_dir) / Path("latest_checkpoint.pth")
    path_to_model: Path = Path(a_cfg.out_dir) / Path("best_model.pth")
    progress_bar = setup_rich_progress(a_cfg.analysis)

    _ = progress_bar.add_task(
        "Epoch",
        total=a_cfg.epoch_count,
        avg_loss=0,
        val_loss=0,
        accuracy_measure=0,
        lr=float(epoch_cfg.optimizer.param_groups[0]["lr"]),
    )
    train_task = progress_bar.add_task("Train", total=len(epoch_cfg.train_loader), avg_loss=0)
    val_task = progress_bar.add_task("Eval", total=len(epoch_cfg.val_loader), avg_loss=0)

    _ = epoch_cfg.model.train()
    loss_sum_val: float = 0.0  # Sum of losses for all samples
    total_samples_for_loss: int = 0  # Denominator for average loss
    avg_loss_train = 0
    avg_loss_val = 0
    metric_val = 0
    labels_tensor: Tensor = torch.empty([1, 1])

    with progress_bar:  # allow for tracking of progress
        for epoch in range(a_cfg.epoch_count):
            for loader in [epoch_cfg.train_loader, epoch_cfg.val_loader]:
                if loader is epoch_cfg.train_loader:
                    # -- Training ----------------------------------------------------------------
                    # ensure model is on proper device
                    progress_bar.reset(
                        train_task, total=len(epoch_cfg.train_loader), train_loss=0, avg_loss=0
                    )
                    train_loss_epoch, labels_tensor, train_total_samples, _ = epoch_loop(
                        a_cfg, epoch_cfg, progress_bar, train_task, "train"
                    )
                    avg_loss_train = train_loss_epoch / train_total_samples
                else:
                    # -- Evaluation Loop ---------------------------------------------------------
                    progress_bar.reset(
                        val_task, total=len(epoch_cfg.val_loader), val_loss=0, avg_loss=0
                    )
                    with torch.no_grad():
                        avg_loss_val, labels_tensor, total_samples_for_loss, metric_val = (
                            epoch_loop(a_cfg, epoch_cfg, progress_bar, val_task, "val")
                        )
                        avg_loss_val = loss_sum_val / total_samples_for_loss

            save_best_model_flag = False
            assert isinstance(metric_val, float)
            if a_cfg.analysis == AnalysisType.REG:
                # Lower MAE is better
                logger.debug(f"At {epoch} / {a_cfg.epoch_count} Val Acc: {metric_val * 100:.2f} %")

                if metric_val < best_val_metric:
                    save_best_model_flag = True
            else:
                # Higher Accuracy is better
                logger.debug(f"At {epoch} / {a_cfg.epoch_count} Val MAE: {metric_val:.9f} µm")

                if metric_val > best_val_metric:
                    best_val_metric = metric_val
                    save_best_model_flag = True

            # Save best model, if metric is better
            if save_best_model_flag:
                # convert to form of 5 numbers, in scientific notation
                evaluation_sci_notation: str = f"{best_val_metric:5e}"
                # create file with name that is unique to evaluation
                best_model_name: str = (
                    path_to_model.name.removesuffix(".pth") + evaluation_sci_notation + ".pth"
                )
                path_to_model_detail: Path = path_to_model.parent / Path(best_model_name)
                _ = torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": epoch_cfg.model.state_dict(),
                        "labels": labels_tensor,
                        "bin_centers": getattr(epoch_cfg.train_loader, "bin_centers", None),
                        "num_bins": a_cfg.num_classes,
                        "val_metric": best_val_metric,
                    },
                    path_to_model_detail,
                )

            # Save latest model, after going through both loaders
            _ = torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": epoch_cfg.model.state_dict(),
                    "bin_centers": getattr(epoch_cfg.train_loader, "bin_centers", None),
                    "num_bins": a_cfg.num_classes,
                    "labels": labels_tensor,
                    "optimizer_state_dict": epoch_cfg.optimizer.state_dict(),
                    "train_loss": avg_loss_train,
                    "val_loss": avg_loss_val,
                    "val_metric": metric_val,
                },
                path_to_checkpoint,
            )

    return avg_loss_train, avg_loss_val, metric_val, best_val_metric


def epoch_loop(
    a_cfg: AutoConfig, epoch_cfg: CoreTrainer, progress_bar: Progress, task_id, type: str
) -> tuple[float, Tensor, int, float | None]:
    """Do one loop of evaluation and training."""
    device = torch.device("cpu") if a_cfg.device == "cpu" else torch.device("cuda")
    if type == "train":
        loader = epoch_cfg.train_loader
        _ = epoch_cfg.model.to(device)
    else:
        loader = epoch_cfg.val_loader
        # reduces memory consumption when doing inference, as is the case for validation
        _ = epoch_cfg.model.eval()

    loss_epoch = 0
    total_samples: int = 0
    abs_err_sum = 0
    metric_val = 0
    total_samples_for_metric = 0  # Denominator for MAE/Accuracy
    labels_tensor: Tensor = torch.empty([1, 1])
    logger.debug(f"Using: {device}, for training.")
    for imgs, labels in loader:
        # sending the images to the tensor is converting them from a PIL image
        # to float32 tensors on [0,1]
        imgs_tens, labels_tens = (
            # PERF: Non-blocking speed up
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        assert (
            next(epoch_cfg.model.parameters()).device == imgs_tens.device == labels_tens.device
        ), (
            f"Images {imgs_tens.device}, labels {labels_tens.device}, or model "
            f"{next(epoch_cfg.model.parameters()).device} not on same device."
        )
        imgs_tens: Tensor
        labels_tens: Tensor

        # -- pass images to model ----------------------------------------------------------------
        # get output of tensor data
        pred: Tensor = epoch_cfg.model(imgs_tens)
        assert pred.dtype == imgs_tens.dtype == torch.float32, (
            "dtype mismatch of predictions and images in training pred: "
            f"{pred.dtype}, imgs: {imgs_tens.dtype}."
        )

        # -- refine preds/labels -----------------------------------------------------------------
        # if we are doing regression and the output tensor is of size [Batch, 1]
        # we need to "squeeze" it into a one dimentisonal tensor of [Batch]
        if a_cfg.analysis == AnalysisType.REG and pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)

        # check loss fn type
        if isinstance(epoch_cfg.loss_fn, nn.BCEWithLogitsLoss):
            labels_tens = labels_tens.float().unsqueeze(0)  # BCE expects float 0/1
            logger.debug(f"labels:, {labels_tens.size}")
        elif isinstance(epoch_cfg.loss_fn, nn.CrossEntropyLoss):
            labels_tens = labels_tens.long()
        else:
            raise Exception(f"Loss function is unknown: {epoch_cfg.loss_fn}")

        # -- calculate loss-----------------------------------------------------------------------
        # this loss is the current average over the batch
        # to find the total loss over the epoch we must sum over
        # each mean loss times the number of images
        loss_fn_current = epoch_cfg.loss_fn(pred, labels_tens)

        # valuation is set to no_grad, this will not work on said tensor
        if type == "train":
            # compute the gradient of the loss
            _ = loss_fn_current.backward()
            # compute one step of the optimization algorithm
            epoch_cfg.optimizer.step()
            # reset gradients each loop
            epoch_cfg.optimizer.zero_grad()

        # sum the value of the loss, scaled to the size of image tensor
        # PERF: huge performance hit <05-26-25>
        loss_epoch += cast("float", (loss_fn_current.item() * imgs_tens.size(0)))
        total_samples += cast("int", imgs_tens.size(0))

        # -- absolute error sum & labels tensor --------------------------------------------------
        if a_cfg.analysis == AnalysisType.REG:
            # conditionally de-normalise as tensors
            z_pred_m: Tensor = pred * epoch_cfg.z_sig.magnitude + epoch_cfg.z_mu.magnitude
            z_true_m: Tensor = labels_tens * epoch_cfg.z_sig.magnitude + epoch_cfg.z_mu.magnitude
            assert z_true_m.size() == z_pred_m.size(), (
                "z_pred and z_true are not the same size, cannot be compared"
            )
            abs_err_sum += torch.sum(torch.abs(z_pred_m - z_true_m)).item()
            total_samples_for_metric += z_true_m.numel()

            # continuous value
            labels_tensor = torch.as_tensor(epoch_cfg.evaluation_metric, dtype=torch.float32)
        else:
            # Get predicted class indices
            pred_classes = torch.argmax(pred, dim=1)  # Shape: [B]
            # Labels size should be [B] and long type
            # Ensure labels are the same shape as pred_classes for comparison
            assert pred_classes.shape == labels_tens.shape, (
                f"Shape mismatch: pred_classes {pred_classes.shape}, labels {labels_tens.shape}"
            )

            # Sum predictions
            # PERF: huge performance hit <05-26-25>
            abs_err_sum += (pred_classes == labels_tens).sum().item()
            # number of correct predictions
            total_samples_for_metric += labels_tens.size(0)

            # using bin value
            labels_tensor = torch.as_tensor(epoch_cfg.evaluation_metric)

        metric_val: float = float(abs_err_sum / total_samples_for_metric)

        # update progress bar
        progress_bar.update(task_id, advance=1, loss=f"{loss_epoch / total_samples:.4f}")

    assert isinstance(loss_epoch, float)
    # evaluation
    if type == "val":
        assert isinstance(metric_val, float)
        return (loss_epoch, labels_tensor, total_samples, metric_val)
    # else train, not returning metric valuation
    return (loss_epoch, labels_tensor, total_samples, None)


def setup_rich_progress(train_type: AnalysisType) -> Progress:
    """Create and configure a Rich Progress bar for training monitoring."""
    metric_col_name = "val_mae" if train_type == AnalysisType.REG else "val_acc"
    metric_col_fmt = "{:.4f}" if train_type == AnalysisType.REG else "{:.2%}"

    return Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",  # Separator
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        RateColumn(),
        "•",
        MetricColumn("train_loss", fmt="{:.4f}", style="magenta"),
        "•",
        MetricColumn("val_loss", fmt="{:.4f}", style="yellow"),
        "•",
        MetricColumn(metric_col_name, fmt=metric_col_fmt, style="green"),
        "•",
        MetricColumn("lr", fmt="{:.1e}", style="dim cyan"),  # Shorter LR format
        SpinnerColumn(),
        transient=False,  # Keep finished tasks visible
    )
