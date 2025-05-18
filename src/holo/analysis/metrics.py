from typing import Literal
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from holo.data.dataset import HologramFocusDataset

# limit the accepted strings
type AnalysisKind = Literal["reg", "cls"]


def gather_z_preds(
    model: Module,
    analysis: AnalysisKind,
    loader: DataLoader[tuple[Tensor, Tensor]],
    dataset: HologramFocusDataset,
    device: torch.device | str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
    z_pred_chunks: list[npt.NDArray[np.float64]] = []
    z_true_chunks: list[npt.NDArray[np.float64]] = []
    x: Tensor
    y: Tensor
    out: Tensor

    with torch.no_grad():
        for x, y in loader:  # y is class index
            # non_blocking corresponds to allowing for multiple tensors to be sent to device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)  # pass in data to model

            # convert z to um from object parameters
            if analysis == "reg":  # float outputs are depth in meters
                z_pred = cast(npt.NDArray[np.float64], (out.squeeze(1) * dataset.z_sigma + dataset.z_mu).cpu().numpy())
                z_tgt = cast(npt.NDArray[np.float64], (y * dataset.z_sigma + dataset.z_mu).cpu().numpy())
            else:  # class -> bin index
                # pick the form that matches network
                if out.ndim == 2:
                    # if model outputs 2D tensor of scores/probabilities, set to 1D
                    cls_pred = out.argmax(dim=1).cpu()
                else:
                    # if model outputs tensor already including an index, "squeeze" out anything but the data
                    cls_pred = out.squeeze().cpu().long()
                    cls_pred = out.argmax(1).cpu()

                cls_pred_num: npt.NDArray[np.int32] = cast(npt.NDArray[np.int32], (cls_pred.numpy()))
                z_pred: npt.NDArray[np.float64] = dataset.bin_centers_m[cls_pred_num]
                z_tgt: npt.NDArray[np.float64] = dataset.bin_centers_m[y.cpu().numpy()]

            # store each of these values
            z_pred_chunks.append(z_pred)
            z_true_chunks.append(z_tgt)

    z_preds = np.concatenate(z_pred_chunks, dtype=np.float32)
    z_true = np.concatenate(z_true_chunks, dtype=np.float32)
    return np.asarray(z_preds, dtype=np.float32).ravel(), np.asarray(z_true, dtype=np.float32).ravel()


def wrap_phase(p: npt.NDArray[np.float32]):
    """One liner to wrap value in pi -> - pi."""
    return (p + np.pi) % (2 * np.pi) - np.pi


def phase_metrics(org_phase: npt.NDArray[np.float32], recon_phase: npt.NDArray[np.float32]):
    """Calculate the mean average error and the phase cosine similarity."""
    diff = wrap_phase(org_phase - recon_phase)
    mae: float = np.abs(diff).mean(dtype=float)
    cos_sim = np.mean(np.cos(diff), dtype=float)  # 1.0 -> perfect match
    return {"MAE_phase": mae, "CosSim": cos_sim}


def error_metric(expected: npt.NDArray[np.float64], observed: npt.NDArray[np.float64], max_px: float):
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
    rmse = cast(np.float64, np.sqrt(mse))
    nrmse = rmse / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = cast(np.float64, 10 * np.log10((max_px**2) / mse))
    return nrmse, psnr
