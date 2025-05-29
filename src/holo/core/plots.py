# TODO: change the documentation around display to propearly describe it <05-19-25, luxShrine >
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.progress import track
from scipy.stats import chi2 as chi2_dist

from holo.core.metrics import wrap_phase
from holo.infra.dataclasses import PlotPred
from holo.infra.util.image_processing import validate_bins
from holo.infra.util.paths import static_root
from holo.infra.util.types import DisplayType

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

logger = logging.getLogger(__name__)  # __name__ is a common choice


# TODO: change the show flag to be comething like mode = "save","show","both" <05-18-25>
def _save_show_plot(
    in_fig: Figure | go.Figure,
    display: DisplayType,
    title: str | None,
    save_path: str = "",
):
    """Help for repeated save or show functionality."""
    print()
    if len(save_path) < 1:
        save_path = static_root().as_posix()
    if title is None:
        title = "unset_title"
    if type(in_fig) is Figure:  # matplotlib
        in_fig.savefig(save_path, dpi=300)
        logger.info(f"{title} plot saved to, {Path(save_path)}")
        if display != DisplayType.SAVE.value:
            logger.info("Displaying matplotlib plot...")
            plt.show()
    elif type(in_fig) is go.Figure:  # plotly
        if display != DisplayType.SAVE.value:
            logger.info("Displaying plot...")
            in_fig.show()
        else:
            logger.info("Saving plotly figure...")
            html_savepath = Path(save_path).with_suffix(".html")
            _ = in_fig.write_html(html_savepath)
            logger.info(f"{title} plot saved to, {html_savepath}")
    else:
        logger.error("Unkown plot type passed to save plot function.")


def plot_actual_versus_predicted(
    plot_pred: PlotPred, title, save_root, display: DisplayType
) -> None:
    """Plot actual vs. predicted values for both training and testing sets for classification."""
    # check for errors to ensure can be plotted
    z_val = np.array(plot_pred.z_test)
    z_train = np.array(plot_pred.z_train)
    z_val_pred = np.array(plot_pred.z_test_pred)
    # __AUTO_GENERATED_PRINT_VAR_START__
    print(
        f"plot_actual_versus_predicted z_val_pred: {str(z_val_pred)}"
    )  # __AUTO_GENERATED_PRINT_VAR_END__
    z_train_pred = np.array(plot_pred.z_train_pred)
    assert z_val.shape == z_val_pred.shape, "z_val is not the same shape as z_val_pred"
    assert z_train.shape == z_train_pred.shape, "z_train is not the same shape as z_train_pred"
    fig: go.Figure = go.Figure()

    # plt.scatter(eval_z_pred_arr, eval_z_true_arr, alpha=0.5, color="yellow", label="Validation")
    # plt.scatter(train_z_pred_arr, train_z_true_arr, alpha=0.1, color="blue", label="Train")
    # plt.legend()
    # plt.xlabel("Predicted Value")
    # plt.ylabel("True Value")
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # plt.title("True vs Predicted Value")
    # plt.savefig("plot.png")

    # global limits
    conc = np.concatenate(
        [z_val_pred, z_val, z_train_pred, z_train]
    )  # combine all values into one array
    span = np.ptp(conc)  # returns range of values "peak to peak"
    # create the limits of the plot so it pads the plotted line
    z_min: float = cast("float", conc.min() - span * 1e-6)
    z_max: float = cast("float", conc.max() + span * 1e-6)

    # if my model was perfect it would match the known dataset values to the predicted values
    # create an ideal line to measure against
    # _ = fig.add_trace(go.Scatter(x=[z_min, z_max], y=[z_min, z_max], mode="lines", name="Ideal"))

    # for the validation values, use a hexbin which shows the density of points in a given region of the plot

    _ = fig.add_trace(go.Scatter(x=z_train, y=z_train_pred, mode="markers", name="train"))
    _ = fig.add_trace(go.Scatter(x=z_val, y=z_val_pred, mode="markers", name="val"))

    if title:
        # x/y axis label, title, grid
        _ = fig.update_layout(
            # yaxis_zeroline=True,
            title_text=title,
            xaxis_title_text="Actual focus depth $(mu m)$",
            yaxis_title_text="Predicted focus depth $(mu m)$",
            legend_title_text="Legend",
            hovermode="closest",
            # xaxis_range=[z_min, z_max],
        )
    # ax.grid(True, linestyle=":")  # type: ignore

    # uncertainty ribbon, better than having tons of error bars
    if plot_pred.zerr_train is None or plot_pred.zerr_test is None:
        logger.warning("No error arrays supplied -> skipping uncertainty ribbon.")
    else:
        min_samples = 8  # minimum values in a bin
        sigma_floor = 1.0  # micro meters, scale ought to vary data to data
        q = 1  # how many sigma to show on plot, this is the width of the error band

        z_bins = np.linspace(z_min, z_max, 50, dtype=np.float32)
        centers = 0.5 * (
            z_bins[:-1] + z_bins[1:]
        )  # find the center of the bins, which is the value at which they appear
        digit = (
            np.digitize(z_train, z_bins) - 1
        )  # find which bin each value falls into, range of 0-28

        good_z_bins = validate_bins(centers, digit, min_samples, z_train_pred, sigma_floor)

        # "unzips" the list of arrays into each subsequent value as a numpy array
        # strict false to avoid raising a value error if arrays are different sizes
        x_train_np, mu_train_np, sigma_train_np = (
            np.asarray(v, dtype=np.float64) for v in zip(*good_z_bins, strict=False)
        )

        # chi squared statistic
        chi2 = np.sum(((mu_train_np - x_train_np) / sigma_train_np) ** 2)  # chi^2 of train_z
        dof = len(x_train_np) - 1  # set dof to the length of the number of measurements
        chi2_red = chi2 / dof
        # p value to measure if my data significantly deviates from expected
        p_val = 1.0 - chi2_dist.cdf(chi2, dof)

        logger.info(f"chi^2 / dof = {chi2:.1f} / {dof} -> chi^2_red = {chi2_red:.2f}")
        logger.info(f"p-value  = {p_val:.3f}")

        # measure the percent of values actually inside the standard deviation band
        sigma_bins = np.full_like(
            centers, np.nan
        )  # initialize an array for SD bins, size of centers
        for count, _, sig in good_z_bins:
            # calculate the difference of actual/pred in a bin, take the minimum value
            # at that index, record the sigma value
            sigma_bins[np.argmin(np.abs(centers - count))] = sig

        digit_val = np.digitize(z_val, z_bins) - 1  # digitize val values
        sigma_val = sigma_bins[digit_val]  # get the SD of each bin of z_val values

        # create a condition such that we only evaluate finite and non-negative standard deviations
        sig_cond = np.isfinite(sigma_val) & (sigma_val > 0)
        # number of hits: the count of predictions that differ from the actual less than q standard deviations
        hits = np.abs(z_val_pred[sig_cond] - z_val[sig_cond]) < q * sigma_val[sig_cond]
        hit_rate = hits.mean() * 100  # return as a percentage

        logger.info(f"Validation MAE  : {np.abs(z_val_pred - z_val).mean():7.2f} µm")
        logger.info(rf"% inside +-{q} sigma ribbon (val): {hit_rate:5.1f}%")

    # NOTE: must create legend after all plots have been created
    # plt.tight_layout()
    assert type(fig) is go.Figure
    _save_show_plot(in_fig=fig, save_path=save_root, display=display, title=title)


def plot_residual_vs_true(plot_info, title, savepath, display):
    """Plot residual vs true depth using Plotly."""
    res_m = plot_info.z_pred_m - plot_info.z_true_m
    n_bins = max(10, len(plot_info.z_true_m) // 50)
    # Ensure bins_m has at least 2 elements for np.linspace and subsequent logic
    bins_m_np = np.linspace(
        plot_info.z_true_m.min(), plot_info.z_true_m.max(), n_bins, dtype=np.float32
    )

    # Calculate running mean & ±σ
    bin_idx = np.digitize(np.array(plot_info.z_true_m), bins_m_np)
    mu_list: list[np.float32] = []
    sd_list: list[np.float32] = []
    xc_list: list[np.float32] = []
    # The loop should go up to len(bins_m_np) to cover all bins defined by linspace.
    # np.digitize with n_bins points creates n_bins-1 intervals.
    # Iterating from 1 to len(bins_m_np) (or n_bins) means checking indices 1 to n_bins-1 based on digitize's output.
    for i in track(range(1, len(bins_m_np)), description="Bin checking (Plotly)..."):
        mask: np.intp = bin_idx == i
        if mask.any():  # at least one sample in the bin
            mu_list.append(res_m[mask].mean())
            sd_list.append(res_m[mask].std())
            xc_list.append(0.5 * (bins_m_np[i] + bins_m_np[i - 1]))

    mu_np = np.array(mu_list, dtype=np.float64)
    sd_np = np.array(sd_list, dtype=np.float32)
    xc_np = np.array(xc_list, dtype=np.float32)

    sig_x = np.concatenate([xc_np, xc_np[::-1]])
    sig_y = np.concatenate([mu_np + sd_np, (mu_np - sd_np)[::-1]], dtype=np.float64)

    fig: go.Figure = go.Figure()

    # Scatter plot of residuals
    _ = fig.add_trace(
        go.Scatter(
            x=plot_info.z_true_m,
            y=res_m,
            mode="markers",
            marker={"size": 6, "opacity": 0.3, "color": "grey"},
            name="Residuals",
        )
    )

    # Running mean line
    if len(xc_np) > 0:  # only plot if there's data
        _ = fig.add_trace(
            go.Scatter(
                x=xc_np,
                y=mu_np,
                mode="lines",
                line={"color": "blue", "width": 2},
                name="Mean",
            )
        )

        # ±1 sigma band
        _ = fig.add_trace(
            go.Scatter(
                x=sig_x,
                y=sig_y,
                fill="toself",
                fillcolor="rgba(0,0,255,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="±1 σ",
            )
        )

    # Horizontal line at y=0
    _ = fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    _ = fig.update_layout(
        title_text=title,
        xaxis_title_text="True focus depth (µm)",
        yaxis_title_text="Residual (pred–true) (µm)",
        legend_title_text="Legend",
        hovermode="closest",
    )

    _save_show_plot(in_fig=fig, save_path=savepath, display=display, title=title)


def plot_violin_depth_bins(
    plot_info,
    title,
    savepath,
    display: DisplayType = DisplayType.SAVE,
):
    # sanity
    z_true_m = np.array(plot_info.z_true_m)
    z_pred_m = np.array(plot_info.z_pred_m)
    assert z_pred_m.shape == z_true_m.shape, "Vectors must match"
    depth_um = z_true_m * 1e6
    err_um = (z_pred_m - z_true_m) * 1e6

    # choose depth bins so each violin has ~50 points
    n_bins = max(10, len(depth_um) // 50)  # tweak divisor as desired
    np.linspace(depth_um.min(), depth_um.max(), n_bins + 1, dtype=np.float64)
    bin_idx = np.digitize(depth_um, plot_info.bins) - 1  # → 0 … n_bins-1

    df = pl.DataFrame(
        {
            "bin": bin_idx,
            "err_um": err_um,
        }
    )

    fig = go.Figure(go.Violin(y=df["err_um"], x=df["bin"], width=0.9))

    _ = fig.update_layout(
        yaxis_zeroline=True,
        title_text=title,
        xaxis_title_text="True focus depth (µm)",
        yaxis_title_text="(pred true) (µm)",
        legend_title_text="Legend",
        hovermode="closest",
    )

    _save_show_plot(in_fig=fig, save_path=savepath, display=display, title=title)


def plot_hexbin_with_marginals(
    plot_info,
    title,
    savepath,
    display: DisplayType = DisplayType.SAVE,
):
    """Plot density of predictions of z."""
    # data
    mask = np.isfinite(np.array(plot_info.z_true_m)) & np.isfinite(np.array(plot_info.z_pred_m))
    z_true_m, z_pred_m = plot_info.z_true_m[mask], plot_info.z_pred_m[mask]

    if z_true_m.size == 0:
        logger.warning("plot_hexbin: no finite points after filtering")
        return

    z_true_um, z_pred_um = z_true_m * 1e6, z_pred_m * 1e6

    # figure & main hexbin
    df = pl.DataFrame(
        {
            "z true": z_true_um,
            "z pred": z_pred_um,
        }
    )
    fig = px.density_heatmap(df, x="z true", y="z pred", marginal_x="histogram")
    assert type(fig) is go.Figure
    _save_show_plot(in_fig=fig, save_path=savepath, display=display, title=title)


def plot_amp_phase(
    amp_recon: npt.NDArray[Any],
    phase_recon: npt.NDArray[Any],
    *,  # allows for parsing in truth values
    amp_true: npt.NDArray[Any] | None = None,
    phase_true: npt.NDArray[Any] | None = None,
    savepath: str = "phase_amp.png",
    display: DisplayType = DisplayType.SHOW,
):
    """Visualise amplitude & phase reconstruction.

    Args:
        amp_recon, phase_recon : arrays
            Results from your Fresnel‑solver.
        amp_true, phase_true : arrays or None
            Provide these **only if** you truly know the ground‑truth field.
            When they are None the function hides GT / error panels.

    """
    title = "amp phase"

    # if ground truth is provided
    if amp_true is not None and phase_true is not None:
        assert amp_true.shape == amp_recon.shape
        assert phase_true.shape == phase_recon.shape

        # figure layout
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        (ax_at, ax_ar, ax_ae, ax_pt, ax_pr, ax_pe) = axes.flatten()
        # force each type to be an axes
        assert type(ax_at) is Axes
        assert type(ax_ar) is Axes
        assert type(ax_ae) is Axes
        assert type(ax_pt) is Axes
        assert type(ax_pr) is Axes
        assert type(ax_pe) is Axes

        # amplitude
        im0: AxesImage = ax_at.imshow(amp_true, cmap="gray")
        _ = ax_at.set_title("Amplitude – ground‑truth")
        _ = fig.colorbar(im0, ax=ax_at, shrink=0.8)

        amp_err = np.abs(amp_true - amp_recon)
        im2: AxesImage = ax_ae.imshow(amp_err, cmap="inferno")
        _ = ax_ae.set_title("Amplitude error")
        _ = fig.colorbar(im2, ax=ax_ae, shrink=0.8)

        im3: AxesImage = ax_pt.imshow(phase_true, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        _ = ax_pt.set_title("Phase – ground‑truth")
        _ = fig.colorbar(im3, ax=ax_pt, shrink=0.8)

        phase_err = wrap_phase(phase_true - phase_recon)
        im5 = ax_pe.imshow(phase_err, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        _ = ax_pe.set_title("Phase error (wrapped)")
        _ = fig.colorbar(im5, ax=ax_pe, shrink=0.8)

    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        (ax_ar, ax_ae, ax_pr, ax_pe) = axes.flatten()  # only 4 axes
        assert type(ax_ar) is Axes
        assert type(ax_ae) is Axes
        assert type(ax_pr) is Axes
        assert type(ax_pe) is Axes

        im1: AxesImage = ax_ar.imshow(amp_recon, cmap="gray")
        _ = ax_ar.set_title("Amplitude – recon")
        _ = fig.colorbar(im1, ax=ax_ar, shrink=0.8)

        # phase
        im4: AxesImage = ax_pr.imshow(phase_recon, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        _ = ax_pr.set_title("Phase – recon")
        _ = fig.colorbar(im4, ax=ax_pr, shrink=0.8)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    _save_show_plot(in_fig=fig, save_path=savepath, display=display, title=title)
