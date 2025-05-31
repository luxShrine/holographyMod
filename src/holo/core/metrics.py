import logging
from typing import Literal, cast

import numpy as np
import torch
from rich.progress import track
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from holo.infra.datamodules import Np1Array32, Np1Array64
from holo.infra.util.types import AnalysisType

logger = logging.getLogger(__name__)


def gather_z_preds(
    model: Module,
    analysis: AnalysisType,
    t_loader: DataLoader[tuple[Tensor, Tensor]],
    v_loader: DataLoader[tuple[Tensor, Tensor]],
    usr_device: Literal["cuda", "cpu"],
    bin_centers_phys: Np1Array64 | None = None,
    z_mu_phys: float | None = None,
    z_sig_phys: float | None = None,
) -> tuple[Np1Array64, Np1Array64, Np1Array64, Np1Array64]:
    """Combine model predictions info format appropriate for comparison.

    Args:
        model (Module): Class of neural network used for prediction.
        analysis (AnalysisType): Type of analysis performed.
        t_loader (DataLoader): Iterable that contains training dataset samples.
        v_loader (DataLoader): Iterable that contains evaluation dataset samples.
        dataset (HologramFocusDataset): Custom dataset object for passing in bin values.
        usr_device (str): Device used for analysis.
        z_mu_phys: Mean of training z_values, physical units for regression.
        z_sig_phys: Std of training z_values, physical units for regression.
        bin_centers_phys: Physical values of bin centers, for classification.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray]: The z-value predictions and truth
        values for each respective loader.

    """
    _: Module = model.eval()  # set model into evaluation rather than training mode
    _ = model.to(usr_device)  # ensure model is on expected device
    logger.info(f"Using: {usr_device}, for recovering Z values.")
    eval_z_pred_list: list[Np1Array64] = []
    eval_z_true_list: list[Np1Array64] = []
    train_z_pred_list: list[Np1Array64] = []
    train_z_true_list: list[Np1Array64] = []

    with torch.no_grad():
        for loader in [t_loader, v_loader]:
            for imgs, labels in track(loader, "Gathering z predictions..."):
                # non_blocking means allowing for multiple tensors to be sent to device
                imgs = imgs.to(usr_device, non_blocking=True)
                labels_device = labels.to(usr_device, non_blocking=True)
                assert next(model.parameters()).device == imgs.device == labels.device, (
                    f"Images {imgs.device}, labels {labels.device}, or model {next(model.parameters()).device} not on same device."
                )

                # pass in data to model
                preds = model(imgs)
                # convert back to physical units
                if analysis == AnalysisType.REG:
                    if z_mu_phys is None or z_sig_phys is None:
                        raise ValueError(
                            "z_mu_phys and z_sig_phys required for regression de-normalization"
                        )

                    # bring predictions back to cpu
                    preds_arr = preds.squeeze().cpu().numpy() * z_sig_phys + z_mu_phys
                    labels_arr = labels.cpu().numpy() * z_sig_phys + z_mu_phys

                elif analysis == AnalysisType.CLASS:
                    if bin_centers_phys is None:
                        raise ValueError("bin_centers_phys required for classificaton conversion")

                    # argmax returns a tensor containing the indices that hold
                    # the maximimum values of the input tensor across the selected
                    # dimension/axis. Here it grabs the indicies of the predictions,
                    # which ought to correspond to the integer bins in the label.
                    preds_arr = preds.argmax(dim=1).unsqueeze(0).cpu().numpy()
                    labels_arr = labels_device.unsqueeze(0).cpu().numpy()

                if loader == t_loader:
                    train_z_pred_list += list(preds_arr[:, 0])
                    train_z_true_list += list(labels_arr[:, 0])
                else:
                    eval_z_pred_list += list(preds_arr[:, 0])
                    eval_z_true_list += list(labels_arr[:, 0])

    # store each of these values
    eval_z_pred_arr: Np1Array64 = np.array(eval_z_pred_list, dtype=np.float64)
    eval_z_true_arr: Np1Array64 = np.array(eval_z_true_list, dtype=np.float64)
    train_z_pred_arr: Np1Array64 = np.array(train_z_pred_list, dtype=np.float64)
    train_z_true_arr: Np1Array64 = np.array(train_z_true_list, dtype=np.float64)

    return (eval_z_pred_arr, eval_z_true_arr, train_z_pred_arr, train_z_true_arr)


def wrap_phase(p: Np1Array32):
    """One liner to wrap value in pi -> - pi."""
    return (p + np.pi) % (2 * np.pi) - np.pi


def phase_metrics(org_phase: Np1Array32, recon_phase: Np1Array32):
    """Calculate the mean average error and the phase cosine similarity."""
    diff = wrap_phase(org_phase - recon_phase)
    mae: float = np.abs(diff).mean(dtype=float)
    cos_sim = np.mean(np.cos(diff), dtype=float)  # 1.0 -> perfect match
    return {"MAE_phase": mae, "CosSim": cos_sim}


def error_metric(expected: Np1Array64, observed: Np1Array64, max_px: float):
    """Find the normalized root mean square error, and the peak noise to signal ratio of two quantities.

    Args:
        expected: Array of image being reconstructed.
        observed: Array of reconstsucted image.
        max_px: The maximum pixel value of an image, e.g. 255 for 8-bit images.

    Returns:
        The NRMSE, and PSNR of these two images.

    """
    # Mean squared error (MSE) -> Peak Signal to noise ratio (PSNR)
    # MSE = 1/n \sum_{i=1}^{n} ( x_i - \hat{x}_{i} )^{2}
    mse = np.mean((expected - observed) ** 2, dtype=np.float64)
    rmse = cast("np.float64", np.sqrt(mse))
    nrmse = rmse / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = cast("np.float64", 10 * np.log10((max_px**2) / mse))
    return nrmse, psnr
