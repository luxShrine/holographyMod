import os
from pathlib import Path
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl
from PIL import Image
from PIL import UnidentifiedImageError
from PIL.Image import Image as ImageType
from rich.pretty import pprint
from torch.utils.data import Dataset

from holo.util.log import logger

_T_co = TypeVar("_T_co", covariant=True)


# TODO: <04-27-25>
class HologramFocusDataset(Dataset[tuple[ImageType, int]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self,
        hologram_dir: Path,
        metadata_csv: str,
        crop_size: int = 512,
        class_steps: float = 10.0,
    ) -> None:
        """Intialize the class by assigning and cleaning values.

        Args:
        hologram_dir: Base directory potentially needed for resolving paths (if not absolute/relative to CSV).
        metadata_csv: Path to CSV mapping filename -> {distance_mm, wavelength_nm}.
        crop_size: output patch size (square).
        class_steps: discretization step for focus classes.

        """
        super().__init__()
        self.crop_size: int = crop_size
        self.class_steps: float = class_steps / 1000  # convert to mm
        self.hologram_dir: Path = hologram_dir  # Store path for hologram directory
        self.metadata_csv_name: str = metadata_csv  # Store metadata csv name

        path_to_csv: Path = self.hologram_dir / metadata_csv
        # read metadata CSV
        unfiltered_df: pl.DataFrame = pl.read_csv(path_to_csv, separator=";")

        def to_abs(rel: str) -> str:
            rel_clean: str = rel.lstrip("./")  # get rid of leading "./"
            return str((self.hologram_dir.parent.expanduser().resolve() / rel_clean).resolve())

        # put the clean paths in the df
        clean_df: pl.DataFrame = unfiltered_df.with_columns(pl.col("path").map_elements(to_abs, pl.String))

        # WARN: debug only
        # pprint(clean_df.item(6, 0))

        # check if row actually points to an existing file
        # filter out non existent paths from df
        path_filtered_df: pl.DataFrame = clean_df.filter(
            pl.col("path").map_elements(lambda p: Path(p).is_file(), pl.Boolean)
        )

        # make sure that there are any files after filtering
        if path_filtered_df.is_empty():
            raise RuntimeError("No hologram files found after filtering non-existing ones")

        # check image is not corrupted
        def _is_valid(path: str) -> bool:
            try:
                with Image.open(path) as im:
                    im.verify()  # quick header check, no full decode
                return True
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                return False

        # check image is not corrupted
        fixed_df = path_filtered_df.filter(pl.col("path").map_elements(_is_valid, pl.Boolean))
        # lists how many images were dropped
        n_bad = path_filtered_df.height - fixed_df.height
        if n_bad:
            logger.warning("Dropped %d corrupt or non-image files", n_bad, extra={"markup": True})

        # Ensure path column is treated as string
        self.records: pl.DataFrame = fixed_df.with_columns(pl.col("path").cast(pl.Utf8))
        # WARN: debug only
        pprint(self.records.slice(100, 10))

        # build class bins
        min_z: float = self.records.select(pl.min("z_value")).item()
        max_z: float = self.records.select(pl.max("z_value")).item()

        # Ensure max_z is included in the bins
        bin_edges: npt.NDArray[np.float64] = np.arange(min_z, max_z + self.class_steps, self.class_steps, float)
        self.num_bins: int = len(bin_edges) - 1
        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # keep to use in plotting function

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[ImageType, int]:
        """Retrieve an item from the dataset by index and return the image along with the object.

        Args:
            idx (int): The index of the item.
            self: The object associated with the item.

        Returns:
            Tuple[Image, int]: A tuple containing the data of the image, and its corresponding index in the total
                               dataset.

        """
        # Using slice and .item() for simplicity here.
        # For performance, consider self.records.row(idx, named=True)
        rec_row = self.records[idx]
        relative_path: str = rec_row["path"].item()  # Extract string value

        # Construct absolute path (assuming path in CSV is relative to CSV location)
        csv_dir = os.path.dirname(self.metadata_csv_name)
        absolute_path = os.path.join(csv_dir, relative_path)

        # load hologram image with PIL
        try:
            img_pil = Image.open(absolute_path)
        except Exception as e:
            pprint(f"Error loading image {absolute_path}: {e}")
            raise  # Re-raise other PIL errors

        # generate class label from z_value
        z_value: float = rec_row["z_value"].item()  # Use .item()

        # TODO: <04-27-25> ?
        # Find which bin z_um falls into (using edges directly)
        # np.digitize returns indices starting from 1. Subtract 1 for 0-based index.
        cls_dig = np.digitize(z_value, self.bin_edges) - 1
        # Clip to ensure label is within [0, num_bins - 1]
        cls = int(np.clip(cls_dig, 0, self.num_bins - 1))  # Ensure integer type

        # Translates pixels through given palette (here RGB) #TODO: again, ought the be gray scale?
        img_pil = Image.open(absolute_path).convert("RGB")

        # TODO: <04-27-25> what is this returning, how does it return the class? the object
        return img_pil, cls
