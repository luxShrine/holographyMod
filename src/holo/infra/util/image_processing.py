# pyright: basic
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image as ImageType

logger = logging.getLogger(__name__)


def crop_center(pil_img: ImageType, crop_width: int, crop_height: int) -> ImageType:
    """Crop provided image into around its center."""
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


def crop_max_square(pil_img: ImageType) -> ImageType:
    """Find the dimensions of image, crop to the largest square around its center."""
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def norm(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize input numpy array."""
    max: np.float64 = np.max(data, keepdims=True)
    min: np.float64 = np.min(data, keepdims=True)
    normed_array = (data - min) / (max - min)
    return normed_array


# check image is not corrupted
def _is_valid(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # quickly check if okay
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def correct_data_csv(csv_path: Path, dataset_path: Path):
    """Ensure input dataframe maps on properly to data and paths."""
    # dataset_parent: str = str(dataset_dir.parent.expanduser().resolve()) + "/"
    dataset_base_path: Path = dataset_path.parent.expanduser().resolve()
    unfiltered_path_df: pl.DataFrame = pl.read_csv(csv_path, separator=";")  # read metadata CSV

    # get each path, get rid of leading "./", prepend it with the path to dataset parent, replace original path column
    # clean_abs_path_df: pl.DataFrame = unfiltered_path_df.with_columns(
    #     pl.col("path").str.replace(r"\./", dataset_parent)
    # )

    abs_path_df: pl.DataFrame = unfiltered_path_df.with_columns(
        pl.col("path")
        .map_elements(lambda p: str(dataset_base_path / Path(p)), return_dtype=pl.Utf8)
        .alias("path")  # Overwrite the original 'path' column
    )

    # check if row actually points to an existing file, must use Path's function, thus map_elements
    filtered_path_df: pl.DataFrame = abs_path_df.filter(
        pl.col("path").map_elements(lambda p: Path(p).is_file(), pl.Boolean)
    )

    # make sure that there are any files after filtering
    if filtered_path_df.is_empty():
        raise RuntimeError("No hologram files found after filtering non-existing ones")

    # check image is not corrupted, again using PIL, thus map_elements
    proper_image_df = filtered_path_df.filter(pl.col("path").map_elements(_is_valid, pl.Boolean))

    # lists how many images were dropped
    n_bad = unfiltered_path_df.height - proper_image_df.height
    if n_bad:
        logger.warning("Dropped %d corrupt or non-image files", n_bad, extra={"markup": True})

    # debug only, print random sample of the dataset to ensure nothing is obviously wrong
    if logger.isEnabledFor(logging.DEBUG):
        with pl.Config(fmt_str_lengths=50):  # make it a little longer for path
            logger.debug(proper_image_df.sample(10, shuffle=True))

    # Ensure path column is treated as string
    # casted_proper_image_df = proper_image_df.with_columns(pl.col("path").cast(pl.Utf8))

    # return casted_proper_image_df
    return proper_image_df


def parse_info(info_file: Path):
    """For each info.txt, extract data from it."""
    info: dict[str, float] = {}
    for line in info_file.read_text().splitlines():
        if not line.strip():
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip("um")  # strip units
        info[key] = float(val)
    return info


def build_metadata_csv(root: Path, out: Path) -> pl.DataFrame:
    """Return a dataframe containing the path to each image file with its data from the info.txt file.

    Args:
        root: directory of dataset
        out: directory to save resulting csv to
    returns:
        dataframe containing dataset paths linked to image's information

    """
    rows: list[dict[str, (str | float)]] = []
    for info in root.rglob("info.txt"):
        meta = parse_info(info)
        for img in info.parent.rglob("*.[jp][np]g"):
            rows.append(
                {
                    "path": str(img.relative_to(root)),
                    **meta,  # unpacks a dictionary into keyword arguments
                }
            )
    df = pl.DataFrame(rows)
    df.glimpse()

    # if output is a directory, save to that directory
    if Path.is_dir(out):
        dir_out = out / "metadata.csv"
        df.write_csv(dir_out, separator=";")
    else:
        df.write_csv(out, separator=";")  # otherwise, save to specified file

    return df


def validate_bins(
    centers: NDArray[np.float64],
    digitized_data: NDArray[np.intp],
    min_samples: int,
    z: NDArray[np.float64],
    sigma_floor: float = 1,
):
    """Check that bins are non-zero, and if so return statistical measures."""
    good_bins: list[
        tuple[float, float, float]
    ] = []  # initialize a list of non-zero bins for x, mu, sigma

    for k, center in enumerate(centers):
        mask = (
            digitized_data == k
        )  # assign only values where the center lines up with values of this particular bin
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


def print_list(input_list: list[Any]) -> None:
    """Small helper to print lists nicely."""
    print(*input_list, sep=",")


def convert_array_tolist_type(data_array: NDArray[Any], data_type: Any):
    """Convert a numpy array to a list of a certain type."""
    data_list: list[Any] = []
    for i in range(len(data_array)):
        try:
            data_list.append(data_type(data_array[i]))
        except ValueError as ve:
            # Handle the exception by raising an error
            logger.exception(f"Cannot convert '{data_array[i]}' to {data_type.__name__}")
            raise ve
        except IndexError as ie:
            logger.exception(f"Cannot find index at '{i}' max index is {len(data_array)}")
            raise ie
    return data_list
