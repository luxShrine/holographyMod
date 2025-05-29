from typing import cast

import numpy as np
import torch
from rich.progress import track
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from holo.infra.datamodules import Np1Array32, Np1Array64
from holo.infra.util.types import AnalysisType


def gather_z_preds(
    model: Module,
    analysis: AnalysisType,
    t_loader: DataLoader[tuple[Tensor, Tensor]],
    v_loader: DataLoader[tuple[Tensor, Tensor]],
    usr_device: torch.device | str,
) -> tuple[Np1Array32, Np1Array32, Np1Array32, Np1Array32]:
    """Combine model predictions info format appropriate for comparison.

    Args:
        model (Module): Class of neural network used for prediction.
        analysis (str): Type of analysis performed (regression "reg" or classification).
        loader (DataLoader): Iterable that contains dataset samples.
        dataset (HologramFocusDataset): Custom dataset object for passing in bin values.
        device (str): Device used for analysis.

    Returns:
        tuple[NDArray, NDArray]: The z-value predictions and truth values.

    """
    _: Module = model.eval()  # set model into evaluation rather than training mode
    eval_z_pred_list: list[Np1Array64] = []
    eval_z_true_list: list[Np1Array64] = []
    train_z_pred_list: list[Np1Array64] = []
    train_z_true_list: list[Np1Array64] = []

    with torch.no_grad():
        for loader in [t_loader, v_loader]:
            for imgs, labels in track(loader, "Gathering z predictions..."):  # y is class index
                # non_blocking corresponds to allowing for multiple tensors to be sent to device
                imgs = imgs.to(usr_device, non_blocking=True)
                labels_device = labels.to(usr_device, non_blocking=True)

                # pass in data to model
                outputs = model(imgs)

                # bring predictions back to cpu
                # argmax returns a tensor containing the indices that hold
                # the maximimum values of the input tensor across the selected
                # dimension/axis. Here it grabs the indicies of the predictions,
                # which ought to correspond to the integer bins in the label.
                outputs_arr = outputs.argmax(dim=1).unsqueeze(0).cpu().numpy()
                labels_arr = labels_device.unsqueeze(0).cpu().numpy()

                if loader == t_loader:
                    train_z_pred_list += list(outputs_arr[:, 0])
                    train_z_true_list += list(labels_arr[:, 0])
                else:
                    eval_z_pred_list += list(outputs_arr[:, 0])
                    eval_z_true_list += list(labels_arr[:, 0])

                # convert z to um from object parameters
                # if analysis == "reg":  # float outputs are depth in meters
                #     # TODO: setup z_mu/sigma
                #
                #     outputs_arr = cast(
                #     "Np1Array64",
                #     (imgs.out.squeeze(1) * dataset.z_sigma + dataset.z_mu).cpu().numpy(),
                # )
                #     z_tgt = cast("Np1Array64", (preds * dataset.z_sigma + dataset.z_mu).cpu().numpy())
                # else:  # abstraction -> physical
                #     # pick the form that matches network
                #     if imgs.ndim == 2:
                #         # if model outputs 2D tensor of scores/probabilities, set to 1D
                #         cls_pred = imgs.argmax(dim=1).cpu()
                #     else:
                #         # if model outputs tensor already including an index,
                #         # "squeeze" out anything but the data
                #         cls_pred = imgs.squeeze().cpu().long()
                #         cls_pred = imgs.argmax(1).cpu()
                #
                #     cls_pred_num: npt.NDArray[np.int32] = cast(
                #         "npt.NDArray[np.int32]", (cls_pred.numpy())
                #     )
                #     edges: npt.ArrayLike = dataset.z_bins
                #     bin_centers_m: npt.NDArray[int] = 0.5 * (edges[:-1] + edges[1:])
                #     z_pred: Np1Array64 = bin_centers_m[cls_pred_num]
                #     z_tgt: Np1Array64 = bin_centers_m[y.cpu().numpy()]

            # store each of these values

    # TODO: convert from bins to physical values <05-26-25>
    eval_z_pred_arr = np.array(eval_z_pred_list)
    eval_z_true_arr = np.array(eval_z_true_list)
    train_z_pred_arr = np.array(train_z_pred_list)
    train_z_true_arr = np.array(train_z_true_list)

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
