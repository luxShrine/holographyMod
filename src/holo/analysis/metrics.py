from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import chi2 as chi2_dist
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from holo.data.dataset import HologramFocusDataset
from holo.util.output import validate_bins
from holo.util.log import logger

# WARN: backend setting, should be temporary fix
# import matplotlib
# matplotlib.use("QtAgg")


def gather_z_preds(
    model: Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    dataset: HologramFocusDataset,
    device: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
    z_preds, z_true = np.array([]), np.array([])

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
            z_preds = np.append(z_preds, z_pred)
            z_true = np.append(z_true, z_tgt)

    # return np.concatenate(z_preds, dtype=np.float64), np.concatenate(z_true, dtype=np.float64)
    return np.asarray(z_preds, dtype=np.float32).ravel(), np.asarray(z_true, dtype=np.float32).ravel()


def _wrap_phase(p: npt.NDArray[np.float32]):
    """One liner to wrap value in pi -> - pi."""
    return (p + np.pi) % (2 * np.pi) - np.pi


def phase_metrics(org_phase: npt.NDArray[np.float32], recon_phase: npt.NDArray[np.float32]):
    """Calculate the mean average error and the phase cosine similarity."""
    diff = _wrap_phase(org_phase - recon_phase)
    mae: float = np.abs(diff).mean()
    cos_sim = np.mean(np.cos(diff), dtype=float)  # 1.0 → perfect match
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
    mse = np.mean((expected - observed) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = 10 * np.log10((max_px**2) / mse)
    return nrmse, psnr


def plot_actual_versus_predicted(
    z_test_pred: npt.NDArray[np.float64],
    z_test: npt.NDArray[np.float64],
    z_train_pred: npt.NDArray[np.float64],
    z_train: npt.NDArray[np.float64],
    yerr_train: npt.NDArray[np.float64] | None = None,
    yerr_test: npt.NDArray[np.float64] | None = None,
    title: None | str = None,
    save_fig: bool = True,
    fname: str = "pred.png",
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Plot actual vs. predicted values for both training and testing sets.

    Args:
        z_test_pred:  Predicted values for the test set.
        z_test:       Actual values for the test set.
        z_train_pred: Predicted values for the train set.
        z_train:      Actual values for the train set.
        yerr_train:   Error in train data.
        yerr_test:    Error in testing data.
        title:        Optional title for the plot.
        save_fig:     If True, save to disk (fname) and close;
                      otherwise, plt.show().
        fname:        Filename to save the figure under.
        figsize:      Figure size in inches (width, height).

    """
    fig, ax = plt.subplots(figsize=figsize)  # create the plot #type: ignore

    # global limits
    conc = np.concatenate([z_test_pred, z_test, z_train_pred, z_train])  # combine all values into one array
    span = np.ptp(conc)  # returns range of values "peak to peak"
    # create the limits of the plot so it pads the plotted line
    vmin: float = conc.min() - span / 4
    vmax: float = conc.max() + span / 4
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # if my model was perfect it would match the known dataset values to the predicted values
    # create an ideal line to measure against
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="Ideal")  # type: ignore

    # plot the train dataset z_value predictions against the known values
    ax.scatter(z_train, z_train_pred, s=6, c="C0", alpha=0.12, rasterized=True)  # type: ignore

    # for the validation values, use a hexbin which shows the density of points in a given region of the plot
    hb = ax.hexbin(z_test, z_test_pred, gridsize=70, cmap="inferno", mincnt=1, bins="log", alpha=0.9, zorder=1)  # type: ignore
    fig.colorbar(hb, ax=ax, label=r"${{ log_{10} }}$(count)")  # colorbar to indicate number of bins #type: ignore

    # x/y axis label, title, grid
    ax.set_xlabel(r"Actual focus depth $(\mu m)$")  # type: ignore
    ax.set_ylabel(r"Predicted focus depth $(\mu m)$")  # type: ignore
    if title:
        ax.set_title(title)  # type: ignore
    ax.grid(True, linestyle=":")  # type: ignore

    # calculate nrmse, ignore the psnr for this
    nrmse, psnr = error_metric(z_test, z_test_pred, 1)
    if np.isinf(psnr):
        print("PSNR infinite (zero MSE)")
    else:
        print(rf"Validation NRMSE : {nrmse:7.2f} µm  PSNR: {psnr}")

    # uncertainty ribbon, better than having tons of error bars
    if yerr_train is None or yerr_test is None:
        logger.warning("No error arrays supplied -> skipping uncertainty ribbon.")
    else:
        min_samples = 16  # minimum values in a bin
        sigma_floor = 1.0  # micro meters, scale ought to vary data to data
        q = 1  # how many sigma to show on plot, this is the width of the error band

        bins = np.linspace(vmin, vmax, 30)  # 29 bins
        centers = 0.5 * (bins[:-1] + bins[1:])  # find the center of the bins, which is the value at which they appear
        digit = np.digitize(z_train, bins) - 1  # find which bin each value falls into, range of 0-28

        good_bins = validate_bins(centers, digit, min_samples, z_train_pred, sigma_floor)

        # "unzips" the list of arrays into each subsequent value as a numpy array
        # strict false to avoid raising a value error if arrays are different sizes
        x_train_np, mu_train_np, sigma_train_np = (
            np.asarray(v, dtype=np.float64) for v in zip(*good_bins, strict=False)
        )

        # chi squared statistic
        chi2 = np.sum(((mu_train_np - x_train_np) / sigma_train_np) ** 2)  # chi^2 of train_z
        dof = len(x_train_np) - 1  # set dof to the length of the number of measurements
        chi2_red = chi2 / dof
        # p value to measure if my data significantly deviates from expected
        p_val = 1.0 - chi2_dist.cdf(chi2, dof)  # type: ignore

        print(f"chi^2 / dof = {chi2:.1f} / {dof} -> chi^2_red = {chi2_red:.2f}")
        print(f"p-value  = {p_val:.3f}")

        # plot the mean value of the z_train predictions, with a band representing the error of the bins
        ax.plot(x_train_np, mu_train_np, color="C0", lw=2, label="Train mean", zorder=4)  # type: ignore
        ax.fill_between(  # type: ignore
            x_train_np,
            mu_train_np - q * sigma_train_np,
            mu_train_np + q * sigma_train_np,
            color="C0",
            alpha=0.20,
            label=rf"${{ \pm }}${q} ${{ \sigma }}$",
            zorder=2,
        )

        # measure the percent of values actually inside the standard deviation band
        sigma_bins = np.full_like(centers, np.nan)  # initialize an array for SD bins, size of centers
        for count, _, sig in good_bins:
            # calculate the difference of actual/pred in a bin, take the minimum value
            # at that index, record the sigma value
            sigma_bins[np.argmin(np.abs(centers - count))] = sig

        digit_val = np.digitize(z_test, bins) - 1  # digitize *test* values
        sigma_val = sigma_bins[digit_val]  # get the SD of each bin of z_test values

        # create a condition such that we only evaluate finite and non-negative standard deviations
        sig_cond = np.isfinite(sigma_val) & (sigma_val > 0)
        # number of hits: the count of predictions that differ from the actual less than q standard deviations
        hits = np.abs(z_test_pred[sig_cond] - z_test[sig_cond]) < q * sigma_val[sig_cond]
        hit_rate = hits.mean() * 100  # return as a percentage

        print(f"Validation MAE  : {np.abs(z_test_pred - z_test).mean():7.2f} µm")
        print(rf"% inside +-{q} sigma ribbon (val): {hit_rate:5.1f}%")

        # Uneeded Residual plot
        # res_val = y_test_pred - y_test
        # ax_res = fig.add_axes([0.13, 0.07, 0.68, 0.18])  # [left, bottom, width, height] # type: ignore
        # ax_res.scatter(y_test, res_val, s=6, alpha=0.4)  # type: ignore
        # ax_res.axhline(0, color="k", lw=1)  # type: ignore
        # ax_res.set_xlabel("Actual (µm)")  # type: ignore
        # ax_res.set_ylabel("Residual")  # type: ignore

    # NOTE: must create legend after all plots have been created
    ax.legend(loc="upper left")  # type: ignore
    plt.tight_layout()
    if save_fig:
        fig.savefig(fname, dpi=300)  # type: ignore
        plt.close(fig)
    else:
        plt.show()  # type: ignore


# hologram_array: Original image, cropped to match reconstruction image.


def plot_amp_phase(
    amp_recon: npt.NDArray[Any],
    phase_recon: npt.NDArray[Any],
    *,
    amp_true: npt.NDArray[Any] | None = None,
    phase_true: npt.NDArray[Any] | None = None,
    savepath: str = "phase_amp.png",
    show: bool = False,
):
    """Visualise amplitude & phase reconstruction.

    Args:
        amp_recon, phase_recon : arrays
            Results from your Fresnel‑solver.
        amp_true, phase_true : arrays or None
            Provide these **only if** you truly know the ground‑truth field.
            When they are None the function hides GT / error panels.

    """
    has_gt = amp_true is not None and phase_true is not None

    if has_gt:
        assert amp_true.shape == amp_recon.shape
        assert phase_true.shape == phase_recon.shape

    # figure layout
    if has_gt:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        (ax_at, ax_ar, ax_ae, ax_pt, ax_pr, ax_pe) = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        (ax_ar, ax_ae, ax_pr, ax_pe) = axes.flatten()  # only 4 axes

    # amplitude
    if has_gt:
        im0 = ax_at.imshow(amp_true, cmap="gray")
        ax_at.set_title("Amplitude – ground‑truth")
        fig.colorbar(im0, ax=ax_at, shrink=0.8)

    im1 = ax_ar.imshow(amp_recon, cmap="gray")
    ax_ar.set_title("Amplitude – recon")
    fig.colorbar(im1, ax=ax_ar, shrink=0.8)

    if has_gt:
        amp_err = np.abs(amp_true - amp_recon)
        im2 = ax_ae.imshow(amp_err, cmap="inferno")
        ax_ae.set_title("Amplitude error")
        fig.colorbar(im2, ax=ax_ae, shrink=0.8)

    # phase
    im4 = ax_pr.imshow(phase_recon, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_pr.set_title("Phase – recon")
    fig.colorbar(im4, ax=ax_pr, shrink=0.8)

    if has_gt:
        im3 = ax_pt.imshow(phase_true, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_pt.set_title("Phase – ground‑truth")
        fig.colorbar(im3, ax=ax_pt, shrink=0.8)

        phase_err = _wrap_phase(phase_true - phase_recon)
        im5 = ax_pe.imshow(phase_err, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_pe.set_title("Phase error (wrapped)")
        fig.colorbar(im5, ax=ax_pe, shrink=0.8)

    # cosmetics
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    fig.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
