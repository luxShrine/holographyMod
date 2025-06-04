import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from rich.progress import track

from holo.core.metrics import error_metric, wrap_phase
from holo.core.optics.reconstruction import torch_recon
from holo.infra.dataclasses import PlotPred
from holo.infra.util.image_processing import crop_max_square, norm
from holo.infra.util.paths import static_root
from holo.infra.util.types import DisplayType

if TYPE_CHECKING:
    from matplotlib.image import AxesImage
    from PIL.Image import Image as ImageType

logger = logging.getLogger(__name__)


def _save_show_plot(
    in_fig: Figure | go.Figure,
    display: DisplayType,
    title: str | None,
    path_to_plot: str = "",
):
    """Help for repeated save or show functionality."""
    if len(path_to_plot) < 1:
        path_to_plot = static_root().as_posix()
    if title is None:
        title = "unset_title"
    if type(in_fig) is Figure:  # matplotlib
        in_fig.savefig(path_to_plot, dpi=300)
        logger.info(f"{title} plot saved to, {Path(path_to_plot)}")
        if display != DisplayType.SAVE.value:
            logger.info("Displaying matplotlib plot...")
            plt.show()
    elif type(in_fig) is go.Figure:  # plotly
        if display != DisplayType.SAVE.value:
            logger.info("Displaying plot...")
            in_fig.show()
        else:
            logger.info("Saving plotly figure...")
            html_savepath = Path(path_to_plot).with_suffix(".html")
            _ = in_fig.write_html(html_savepath)
            logger.info(f"{title} plot saved to, {html_savepath}")
    else:
        logger.error("Unkown plot type passed to save plot function.")


def plot_actual_versus_predicted(plot_info: PlotPred, title: str, path_to_plot: str) -> None:
    """Plot actual vs. predicted values and a confusion matrix for classification."""
    import plotly.figure_factory as ff
    import polars as pl
    from plotly.subplots import make_subplots
    from polars import DataFrame
    from scipy.stats import gaussian_kde
    from sklearn.metrics import confusion_matrix

    z_val_phys = np.array(plot_info.z_test, dtype=np.float64)
    z_train_phys = np.array(plot_info.z_train, dtype=np.float64)
    z_val_pred_phys = np.array(plot_info.z_test_pred, dtype=np.float64)
    z_train_pred_phys = np.array(plot_info.z_train_pred, dtype=np.float64)
    z_val_pred_min = z_val_pred_phys.min()
    z_train_pred_min = z_train_pred_phys.min()
    data = {
        "z_val_phys": z_val_phys,
        "z_train_phys": z_train_phys,
        "z_val_pred_phys": z_val_pred_phys,
        "z_train_pred_phys": z_train_pred_phys,
    }

    # Convert to µm for plotting, concat creates null values automatically since
    # prediction and evaluation arrays aren't the same size
    df: DataFrame = pl.concat(
        items=[pl.DataFrame({name: (val * 1e6)}) for name, val in data.items()],
        how="horizontal",
    )
    # even if some values are null (due to above) it will not be passed into the
    # plot as long as there are values in each bin
    residual_val = pl.col("z_val_pred_phys") - pl.col("z_val_phys")
    residual_train = pl.col("z_train_pred_phys") - pl.col("z_train_phys")
    df: DataFrame = df.with_columns((residual_val).alias("residual_val"))
    df: DataFrame = df.with_columns((residual_train).alias("residual_train"))
    # libraries like scipy and numpy dont play nicely with null values,
    # separate df for numerical manipulation
    df_filled: DataFrame = df.fill_null(0)

    # -- Confusion Matrix plot -------------------------------------------------------------------
    # Ensure bin_edges is not None and is a numpy array for classification
    if plot_info.bin_edges is None:
        raise Exception("bin_edges not provided in PlotPred. Plotting failed.")
    bin_edges = np.array(plot_info.bin_edges, dtype=np.float64)
    assert isinstance(bin_edges[0], np.float64), (
        f"bin_edges contains something other than np.float64, found {type(bin_edges[0])}"
    )

    # Convert physical values back to bin indices for the confusion matrix
    # We'll use the validation set for the confusion matrix as a common practice
    true_indices_val = np.clip(np.digitize(z_val_phys, bin_edges), 1, len(bin_edges) - 1) - 1
    pred_indices_val = np.clip(np.digitize(z_val_pred_phys, bin_edges), 1, len(bin_edges) - 1) - 1

    num_classes = len(bin_edges) - 1

    # Class labels for the confusion matrix axes
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Convert to µm for readability if original units are meters
    class_labels_um = [f"{c * 1e6:.1f}µm" for c in bin_centers]

    cm = confusion_matrix(true_indices_val, pred_indices_val, labels=list(range(num_classes)))

    # Create annotated heatmap for confusion matrix using Plotly Figure Factory
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=class_labels_um,
        y=class_labels_um,
        colorscale="Blues",
        showscale=True,  # Shows the color bar
        reversescale=False,
    )
    _ = fig_cm.update_layout(
        title_text="Confusion Matrix (Validation Set)",
        xaxis_title_text="Predicted Label (Physical Bin Center)",
        yaxis_title_text="True Label (Physical Bin Center)",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(num_classes)),
            "ticktext": class_labels_um,
            "side": "bottom",
        },
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(num_classes)),
            "ticktext": class_labels_um,
            "autorange": "reversed",
        },  # Reversed to match typical CM layout
    )
    # Save or show the confusion matrix
    cm_save_path = Path(path_to_plot).parent / (Path(path_to_plot).stem + "_confusion_matrix.html")
    _save_show_plot(fig_cm, plot_info.display, "Confusion Matrix", cm_save_path.as_posix())

    # -- Scatter plot ----------------------------------------------------------------------------

    # scipy estimates the probability density function of random points using
    # Gaussian kernels. This is generated and then applied to the stacked
    # arrays: vstack([N], [B]) -> [[N], [B]]. Outputing continious value to
    # use for density
    kde_val = gaussian_kde(
        np.vstack(
            [
                df_filled["z_val_pred_phys"].to_numpy().clip(min=z_val_pred_min),
                df_filled["residual_val"].to_numpy(),
            ]
        )
    )
    kde_train = gaussian_kde(
        np.vstack(
            [
                df_filled["z_train_pred_phys"].to_numpy().clip(min=z_train_pred_min),
                df_filled["residual_train"].to_numpy(),
            ]
        )
    )
    density_val = kde_val(
        np.vstack(
            [
                df_filled["z_val_pred_phys"].to_numpy().clip(min=z_val_pred_min),
                df_filled["residual_val"].to_numpy(),
            ]
        )
    )
    density_train = kde_train(
        np.vstack(
            [
                df_filled["z_train_pred_phys"].to_numpy().clip(min=z_train_pred_min),
                df_filled["residual_train"].to_numpy(),
            ]
        )
    )

    # -- create (1×2) layout ---------------------------------------------------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.80, 0.20],  # 80 % to the scatter | 20 % to the histogram
        shared_yaxes=True,  # axes line up
        horizontal_spacing=0.04,
    )
    # -- main scatter coloured by density --------------------------------------------------------
    # Use the non filled df as plotly will properly drop the null values
    _ = fig.add_trace(
        go.Scatter(
            x=df["z_val_pred_phys"].to_numpy(),
            y=df["residual_val"].to_numpy(),
            mode="markers",
            marker={
                "color": density_val,  # continuous array
                "colorscale": "burg",
                "showscale": True,
                "colorbar": {"title": "val density", "x": 1},  # separate the two scales
                "size": 9,
                "opacity": 0.9,
            },
            name="Validation",  # legend text
            legendgroup="VAL",  # groups both traces
            showlegend=True,  # visible item in legend
        ),
        row=1,
        col=1,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=df["z_train_pred_phys"].to_numpy(),
            y=df["residual_train"].to_numpy(),
            mode="markers",
            marker={
                "color": density_train,
                "colorscale": "oryel",
                "showscale": True,
                "colorbar": {"title": "train density", "x": 1.1},
                "size": 9,
                "opacity": 0.3,
            },
            name="Train",
            legendgroup="TRAIN",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # -- add the marginal histogram (horizontal for y-axis) --------------------------------------
    _ = fig.add_trace(
        go.Histogram(
            y=df["residual_val"],
            marker_color="rgba(68,1,84,0.7)",
            name="Validation",
            legendgroup="VAL",  # same group -> toggles together
            showlegend=False,  # no second item
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    _ = fig.add_trace(
        go.Histogram(
            y=df["residual_train"],
            marker_color="rgba(255,183,76,0.7)",
            name="Train",
            legendgroup="TRAIN",
            showlegend=False,
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    # --Finish ----------------------------------------------------------------------------------
    # Scatter
    _ = fig.update_xaxes(title="Focus Depth (µm)", row=1, col=1)
    _ = fig.update_yaxes(title="Residual (µm)", row=1, col=1)
    # Histogram region
    _ = fig.update_xaxes(showticklabels=True, row=1, col=2)

    _ = fig.update_layout(
        # template="plotly_white",
        title_text=title,
        legend_title_text="Legend",
        hovermode="closest",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )
    _save_show_plot(fig, plot_info.display, title, path_to_plot)


def plot_residual_vs_true(plot_info, title, path_to_plot):
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

    _save_show_plot(in_fig=fig, path_to_plot=path_to_plot, display=plot_info.display, title=title)


def plot_violin_depth_bins(
    plot_info,
    title,
    path_to_plot,
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

    _save_show_plot(in_fig=fig, path_to_plot=path_to_plot, display=plot_info.display, title=title)


def plot_hexbin_with_marginals(
    plot_info,
    title,
    path_to_plot,
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
    _save_show_plot(in_fig=fig, path_to_plot=path_to_plot, display=plot_info.display, title=title)


def plot_amp_phase(
    img_file_path,
    wavelength,
    ckpt_file,
    crop_size,
    z,
    backbone,
    dx,
    *,  # allows for parsing in truth values
    amp_true: npt.NDArray[Any] | None = None,
    phase_true: npt.NDArray[Any] | None = None,
    path_to_plot: str = "phase_amp.png",
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
    amp_recon: npt.NDArray[Any]
    phase_recon: npt.NDArray[Any]

    image: ImageType = Image.open(img_file_path).convert("RGB")
    holo_org = np.array(crop_max_square(image))  # crop, and convert to array
    recon, amp_recon, phase_recon = torch_recon(
        img_file_path, wavelength, str(ckpt_file), crop_size, z, backbone, dx
    )
    n_org = norm(holo_org)
    n_recon = norm(recon)
    nrmsd, psnr = error_metric(n_org, n_recon, 255)
    logger.info(f"the psnr is {psnr} with nrmsd: {nrmsd}")

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
    _save_show_plot(in_fig=fig, path_to_plot=path_to_plot, display=display, title=title)
