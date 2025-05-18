"""Various output helper functions."""

import numpy as np
import numpy.typing as npt

from holo.util.log import logger


def validate_bins(
    centers: npt.NDArray[np.float64],
    digitized_data: npt.NDArray[np.intp],
    min_samples: int,
    z: npt.NDArray[np.float64],
    sigma_floor: float = 1,
):
    """Check that bins are non-zero, and if so return statistical measures."""
    good_bins: list[tuple[float, float, float]] = []  # initialize a list of non-zero bins for x, mu, sigma

    for k, center in enumerate(centers):
        mask = digitized_data == k  # assign only values where the center lines up with values of this particular bin
        n = mask.sum()  # sum to find the total samples in the bin
        if n < min_samples:
            continue  # skip thinly populated bins

        vals = z[mask]  # if there are sufficient values, we can use it
        mu = vals.mean()  # find the average value in this bin
        # find the standard deviation, but limit it such that we don't violate significant figures
        sigma = max(vals.std(ddof=1), sigma_floor)

        good_bins.append((center, mu, sigma))  # store these values for plotting

    # prevent analyzing too few bins
    if len(good_bins) < 2:
        logger.warning("Not enough populated bins to compute chi^2.")
        return good_bins
    else:
        return good_bins
