from pathlib import Path
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from rich.progress import track
from scipy.stats import chi2 as chi2_dist

from holo.analysis.metrics import error_metric
from holo.analysis.metrics import wrap_phase
from holo.util.log import logger
from holo.util.output import validate_bins
from holo.util.paths import static_root


def _save_show_plot(in_fig: Figure | go.Figure, show: bool, title: str | None, save_path: str = ""):
    """Help for repeated save or show functionality."""
    if len(save_path) < 1:
        save_path = static_root().as_posix()
    if title is None:
        title = "unset_title"
    if type(in_fig) is Figure:  # matplotlib
        in_fig.savefig(save_path, dpi=300)
        logger.info(f"{title} plot saved to, {Path(save_path)}")
        if show:
            logger.info("Displaying matplotlib plot...")
            plt.show()  # type: ignore
        else:
            plt.close(in_fig)
    elif type(in_fig) is go.Figure:  # plotly
        if show:
            logger.info("Displaying plot...")
            in_fig.show()  # type: ignore
        logger.info("Saving plotly figure...")
        try:
            save_p = Path(save_path)
            _ = in_fig.write_image(save_p, width=800, height=500)  # Specify dimensions if needed
            logger.info(f"{title} plot saved to, {Path(save_path)}")
        except ValueError:
            try:
                html_savepath = Path(save_path).with_suffix(".html")
                _ = in_fig.write_html(html_savepath)
                logger.warning(
                    "Plotly figure could not be saved as image",
                    f"(Kaleido not installed). Saved as HTML to {html_savepath}",
                )
                logger.info(f"{title} plot saved to, {html_savepath}")
            except Exception as e_html:
                logger.warning(f"Failed to save Plotly figure as HTML: {e_html}")
    else:
        logger.error("Unkown plot type passed to save plot function.")


def plot_actual_versus_predicted(
    z_test_pred: npt.NDArray[np.float64],
    z_test: npt.NDArray[np.float64],
    z_train_pred: npt.NDArray[np.float64],
    z_train: npt.NDArray[np.float64],
    zerr_train: npt.NDArray[np.float64] | None = None,
    zerr_test: npt.NDArray[np.float64] | None = None,
    title: None | str = None,
    save_fig: bool = True,
    # TODO:  change fname to path, fname really ought to be title?
    fname: str = "pred.png",
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """Plot actual vs. predicted values for both training and testing sets for classification.

    Args:
        z_test_pred:  Predicted values for the test set.
        z_test:       Actual values for the test set.
        z_train_pred: Predicted values for the train set.
        z_train:      Actual values for the train set.
        zerr_train:   Error in train data.
        zerr_test:    Error in testing data.
        title:        Optional title for the plot.
        save_fig:     If True, save to disk (fname) and close;
                      otherwise, plt.show().
        fname:        Filename to save the figure under.
        figsize:      Figure size in inches (width, height).

    """
    # check for errors to ensure can be plotted
    assert z_test.shape == z_test_pred.shape, "z_test is not the same shape as z_test_pred"
    assert z_train.shape == z_train_pred.shape, "z_train is not the same shape as z_train_pred"
    fig: go.Figure = go.Figure()

    # global limits
    conc = np.concatenate([z_test_pred, z_test, z_train_pred, z_train])  # combine all values into one array
    span = np.ptp(conc)  # returns range of values "peak to peak"
    # create the limits of the plot so it pads the plotted line
    vmin: float = cast(float, conc.min() - span / 4)
    vmax: float = cast(float, conc.max() + span / 4)

    # if my model was perfect it would match the known dataset values to the predicted values
    # create an ideal line to measure against
    _ = fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", name="Ideal"))
    # _ = ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="Ideal")  # type: ignore

    # plot the train dataset z_value predictions against the known values
    # _ = fig.add_trace(go.Scatter(x=z_train, y=z_train_pred, mode="markers", name="Actual vs. Predictions for Training"))
    # _ = ax.scatter(z_train, z_train_pred, s=6, c="C0", alpha=0.05, rasterized=True)  # type: ignore

    # for the validation values, use a hexbin which shows the density of points in a given region of the plot

    _ = fig.add_trace(go.Scatter(x=z_test, y=z_test_pred, mode="markers", name="hexbin"))
    # hb = ax.hexbin(z_test, z_test_pred, gridsize=70, cmap="inferno", mincnt=1, bins="log", alpha=0.9, zorder=1)  # type: ignore
    # _ = fig.colorbar(hb, ax=ax, label=r"${{ log_{10} }}$(count)")  # colorbar to indicate number of bins #type: ignore

    # _ = ax.set_xlabel(r"Actual focus depth $(\mu m)$")  # type: ignore
    # _ = ax.set_ylabel(r"Predicted focus depth $(\mu m)$")  # type: ignore
    if title:
        # x/y axis label, title, grid
        _ = fig.update_layout(
            # yaxis_zeroline=True,
            title_text=title,
            xaxis_title_text="Actual focus depth $(\mu m)$",
            yaxis_title_text="Predicted focus depth $(\mu m)$",
            legend_title_text="Legend",
            hovermode="closest",
        )
    # ax.grid(True, linestyle=":")  # type: ignore

    # calculate nrmse, ignore the psnr for this
    nrmse, psnr = error_metric(z_test, z_test_pred, 1)
    if np.isinf(psnr):
        logger.info("PSNR infinite (zero MSE)")
    else:
        logger.info(rf"Validation NRMSE : {nrmse:7.2f} µm  PSNR: {psnr}")

    # uncertainty ribbon, better than having tons of error bars
    if zerr_train is None or zerr_test is None:
        logger.warning("No error arrays supplied -> skipping uncertainty ribbon.")
    else:
        min_samples = 8  # minimum values in a bin
        sigma_floor = 1.0  # micro meters, scale ought to vary data to data
        q = 1  # how many sigma to show on plot, this is the width of the error band

        bins = np.linspace(vmin, vmax, 20)
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

        logger.info(f"chi^2 / dof = {chi2:.1f} / {dof} -> chi^2_red = {chi2_red:.2f}")
        logger.info(f"p-value  = {p_val:.3f}")

        # plot the mean value of the z_train predictions, with a band representing the error of the bins
        # sig_positive = mu_train_np + q * sigma_train_np
        # sig_negative = mu_train_np - q * sigma_train_np
        # fig2 = go.Figure()
        # fig2.add_trace(go.scatter(x_train_np, sig_positive))
        # fig2.add_trace(go.scatter(x_train_np, sig_negative))
        # # _ = fig2.add_trace(go.scatter(
        # #         x=x_train_np,
        # #         y=mu_train_np + q * sigma_train_np,
        # #         fill="tonexty",
        # #         # label=rf"${{ \pm }}${q} ${{ \sigma }}$",
        # #     )
        # # )
        #
        # fig2.add_trace(go.scatter(x_train_np, mu_train_np))  # type: ignore
        #
        # _ = fig2.update_layout(
        #     # yaxis_zeroline=True,
        #     title_text="Train mean, with Error Band",
        #     xaxis_title_text="Actual focus depth $(\mu m)$",
        #     yaxis_title_text="Predicted focus depth $(\mu m)$",
        #     legend_title_text="Legend",
        #     hovermode="closest",
        # )

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

        logger.info(f"Validation MAE  : {np.abs(z_test_pred - z_test).mean():7.2f} µm")
        logger.info(rf"% inside +-{q} sigma ribbon (val): {hit_rate:5.1f}%")

    # NOTE: must create legend after all plots have been created
    # plt.tight_layout()
    assert type(fig) is go.Figure
    # assert type(fig2) is go.Figure
    _save_show_plot(in_fig=fig, save_path=fname, show=save_fig, title=title)
    # _save_show_plot(in_fig=fig2, save_path="band", show=save_fig, title="Train mean")


def plot_residual_vs_true(
    z_pred_m: npt.NDArray[np.float64],
    z_true_m: npt.NDArray[np.float64],
    title: str = "Residual vs True depth (Plotly)",
    savepath: str = "residual_plotly.png",
    show: bool = False,
):
    """Plot residual vs true depth using Plotly."""
    res_m = z_pred_m - z_true_m
    n_bins = max(10, len(z_true_m) // 50)
    # Ensure bins_m has at least 2 elements for np.linspace and subsequent logic
    bins_m_np = np.linspace(z_true_m.min(), z_true_m.max(), n_bins, dtype=np.float32)

    # Calculate running mean & ±σ
    bin_idx = np.digitize(z_true_m, bins_m_np)
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
            x=z_true_m, y=res_m, mode="markers", marker=dict(size=6, opacity=0.3, color="grey"), name="Residuals"
        )
    )

    # Running mean line
    if len(xc_np) > 0:  # only plot if there's data
        _ = fig.add_trace(go.Scatter(x=xc_np, y=mu_np, mode="lines", line=dict(color="blue", width=2), name="Mean"))

        # ±1 sigma band
        _ = fig.add_trace(
            go.Scatter(
                x=sig_x,
                y=sig_y,
                fill="toself",
                fillcolor="rgba(0,0,255,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
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

    _save_show_plot(in_fig=fig, save_path=savepath, show=show, title=title)


def plot_violin_depth_bins(
    z_pred_m: npt.NDArray[np.float64],
    z_true_m: npt.NDArray[np.float64],
    title: str = "Signed error distribution per depth slice",
    savepath: str = "depth_bins.png",
    show: bool = False,
):
    # sanity
    assert z_pred_m.shape == z_true_m.shape, "Vectors must match"
    depth_um = z_true_m * 1e6
    err_um = (z_pred_m - z_true_m) * 1e6

    # choose depth bins so each violin has ~50 points
    n_bins = max(10, len(depth_um) // 50)  # tweak divisor as desired
    bins = np.linspace(depth_um.min(), depth_um.max(), n_bins + 1, dtype=np.float64)
    bin_idx = np.digitize(depth_um, bins) - 1  # → 0 … n_bins-1
    # bin_cent = 0.5 * (bins[:-1] + bins[1:])  # for labels

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

    _save_show_plot(in_fig=fig, save_path=savepath, show=show, title=title)


def plot_hexbin_with_marginals(
    z_pred_m: npt.NDArray[np.float64],
    z_true_m: npt.NDArray[np.float64],
    title: str = "Prediction density (val)",
    savepath: str = "phase_amp.png",
    show: bool = False,
):
    """Plot density of predictions of z."""
    # data
    mask = np.isfinite(z_true_m) & np.isfinite(z_pred_m)
    z_true_m, z_pred_m = z_true_m[mask], z_pred_m[mask]

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
    _save_show_plot(in_fig=fig, save_path=savepath, show=show, title=title)


def plot_amp_phase(
    amp_recon: npt.NDArray[Any],
    phase_recon: npt.NDArray[Any],
    *,  # allows for parsing in truth values
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
    _save_show_plot(in_fig=fig, save_path=savepath, show=show, title=title)
