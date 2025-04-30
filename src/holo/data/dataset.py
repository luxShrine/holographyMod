import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from PIL import Image
from PIL.Image import Image as ImageType
from rich.pretty import pprint
from torch.utils.data import Dataset

from holo.io.metadata import correct_data_csv
from holo.util.log import logger


# TODO: <04-27-25>
class HologramFocusDataset(Dataset[tuple[ImageType, int]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self,
        hologram_dir: Path,
        metadata_csv: str,
        crop_size: int = 512,
        class_steps: float = 10,
    ) -> None:
        """Assign properties to hologram dataset.

        Args:
        hologram_dir (Path): Base directory used to find relevant paths.
        metadata_csv (str): Path to CSV that maps images to properties, z distance and wavelength.
        crop_size (int): What size to crop image to.
        class_steps (float): How many bins to make in process of making discrete z values.

        """
        # my hologram dataset is a subclass of the torch.utils.data Dataset, this allows initializing both classes
        super().__init__()
        self.crop_size: int = crop_size  # assign the crop size
        self.class_steps: float = class_steps / 1000  # convert to mm
        self.hologram_dir: Path = hologram_dir  # Store path for hologram directory
        self.metadata_csv_name: str = metadata_csv  # Store metadata csv name

        path_to_csv: Path = self.hologram_dir / metadata_csv
        unfiltered_df: pl.DataFrame = pl.read_csv(path_to_csv, separator=";")  # read metadata CSV
        self.records: pl.DataFrame = correct_data_csv(path_to_csv, hologram_dir)

        # lists how many images were dropped
        n_bad = unfiltered_df.height - self.records.height
        if n_bad:
            logger.warning("Dropped %d corrupt or non-image files", n_bad, extra={"markup": True})

        # WARN: debug only
        pprint(self.records.sample(10, shuffle=True))

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
