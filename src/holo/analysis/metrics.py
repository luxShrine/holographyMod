from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes as AxesType
from PIL import Image
from PIL.Image import Image as ImageType
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from holo.data.dataset import HologramFocusDataset


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
    square_error = (np.mean(expected) - np.mean(observed)) ** 2
    mse = np.mean(square_error)
    rmse = np.sqrt(mse)
    nrmse = rmse / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = 10 * np.log10((max_px**2) / mse)
    return nrmse, psnr


def plot_actual_versus_predicted(
    y_test_pred: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    y_train_pred: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    yerr_train: npt.NDArray[np.float64] | None = None,
    yerr_test: npt.NDArray[np.float64] | None = None,
    title: None | str = None,
    save_fig: bool = False,
    fname: str = "pred.png",
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Plot actual vs. predicted values for both training and testing sets.

    Args:
        y_test_pred:  Predicted values for the test set.
        y_test:       Actual values for the test set.
        y_train_pred: Predicted values for the train set.
        y_train:      Actual values for the train set.
        title:        Optional title for the plot.
        save_fig:     If True, save to disk (fname) and close;
                      otherwise, plt.show().
        fname:        Filename to save the figure under.
        figsize:      Figure size in inches (width, height).

    """
    # Concatenate to find global plot limits
    conc = np.concatenate([y_test_pred, y_test, y_train_pred, y_train])
    span = np.ptp(conc)
    vmin = conc.min() - span
    vmax = conc.max() + span

    fig, ax = plt.subplots(figsize=figsize)  # type: ignore
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # train vs test
    # ax.errorbar(y_train, y_train_pred, fmt=",", alpha=0.5, ms=3, ecolor="lightgray", label="Train", capsize=8)  # type: ignore
    # ax.errorbar(y_test, y_test_pred, fmt=",", alpha=0.7, ms=3, ecolor="lightgray", label="Test", capsize=8)  # type: ignore

    ax.errorbar(  # type: ignore
        y_train,
        y_train_pred,
        yerr=yerr_train,
        fmt="o",
        alpha=0.4,
        ms=8,
        ecolor="lightgray",
        capsize=2,
        label="Train",
    )
    ax.errorbar(  # type: ignore
        y_test,
        y_test_pred,
        yerr=yerr_test,
        fmt="s",
        alpha=0.5,
        ms=8,
        ecolor="lightgray",
        capsize=2,
        label="Test",
    )

    # perfect‐prediction line
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=2, label="Ideal")  # type: ignore

    ax.set_xlabel("Actual")  # type: ignore
    ax.set_ylabel("Predicted")  # type: ignore
    if title:
        ax.set_title(title)  # type: ignore
    ax.legend(loc="upper left")  # type: ignore
    ax.grid(True, linestyle=":")  # type: ignore

    if save_fig:
        plt.tight_layout()
        fig.savefig(fname, dpi=300)  # type: ignore
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()  # type: ignore


def plot_amp_phase(
    original_img: str,
    fresnel_amp_img: npt.NDArray[np.float64],
    fresnel_phase_img: npt.NDArray[np.float64],
    nrmsd: np.float64,
    psnr: np.float64,
):
    # temp plotting to see it works

    fig, axs = plt.subplots(2, 2)  # type: ignore
    fig.text(0.5, 0.025, f"$PSNR={psnr}$ \n $NRMS={nrmsd}$")  # type: ignore

    plot_amp: AxesType = axs[0, 0].imshow(fresnel_amp_img, cmap="gray")
    axs[0, 0].set_title("Reconstructed Amplitude")
    fig.colorbar(plot_amp, ax=axs[0, 0])  # type: ignore

    plot_phase: AxesType = axs[0, 1].imshow(fresnel_phase_img, cmap="gray")
    axs[0, 1].set_title("Reconstructed Phase")
    fig.colorbar(plot_phase, ax=axs[0, 1])  # type: ignore

    hologram: ImageType = Image.open(original_img)
    plot_holo: AxesType = axs[1, 0].imshow(hologram, cmap="gray")
    axs[1, 0].set_title("Original Image Phase")
    fig.colorbar(plot_holo, ax=axs[1, 0])  # type: ignore

    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()  # type: ignore
    plt.savefig("amp_phase.png")  # type: ignore
