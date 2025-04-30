import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def error_metric(expected: float, observed: float, max_px: float):
    # Mean squared error (MSE) -> Peak Signal to noise ratio (PSNR)
    # MSE = 1/n \sum_{i=1}^{n} ( x_i - \hat{x}_{i} )^{2}
    square_error = (expected - observed) ** 2
    mse = np.mean(square_error)
    rmsd = np.sqrt(mse)
    nrmsd = rmsd / np.mean(observed)

    # PSNR = 10 log( Max / MSE )
    # MAX = the maximum possible pixel value (255) for 8bit
    psnr = 10 * np.log10((max_px**2) / mse)
    return nrmsd, psnr


def plot_actual_versus_predicted(
    y_test_pred: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
    y_train_pred: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
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
    # mu, sigma = conc.mean(), conc.std()
    # vmin = conc.min() - sigma
    # vmax = conc.max() + sigma
    span = np.ptp(conc)
    vmin = conc.min() - span
    vmax = conc.max() + span

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # train vs test
    ax.errorbar(y_train, y_train_pred, fmt="o", alpha=0.5, ms=8, ecolor="lightgray", label="Train", capsize=2)
    ax.errorbar(y_test, y_test_pred, fmt="s", alpha=0.7, ms=8, ecolor="lightgray", label="Test", capsize=2)

    # perfect‐prediction line
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=2, label="Ideal")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":")

    if save_fig:
        plt.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
