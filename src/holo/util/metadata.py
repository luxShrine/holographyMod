import logging
from pathlib import Path

import polars as pl
from PIL import Image
from PIL import UnidentifiedImageError

from holo.util.log import logger


# check image is not corrupted
def _is_valid(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # quickly check if okay
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def correct_data_csv(path_csv_str: Path, dataset_dir: Path):
    """Ensure input dataframe maps on properly to data and paths."""
    # dataset_parent: str = str(dataset_dir.parent.expanduser().resolve()) + "/"
    dataset_base_path: Path = dataset_dir.parent.expanduser().resolve()
    unfiltered_path_df: pl.DataFrame = pl.read_csv(path_csv_str, separator=";")  # read metadata CSV

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
