from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from PIL import Image
from PIL.Image import Image as ImageType
from rich.pretty import pprint
from torch.utils.data import Dataset

from holo.io.metadata import correct_data_csv
from holo.util.log import logger


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
        self.metadata_csv_path_str: str = metadata_csv  # Store metadata csv name

        path_to_csv: Path = self.hologram_dir / self.metadata_csv_path_str
        unfiltered_df: pl.DataFrame = pl.read_csv(path_to_csv, separator=";")  # read metadata CSV
        # cleanup data remove bad images, bad paths
        self.records: pl.DataFrame = correct_data_csv(path_to_csv, self.hologram_dir)

        # lists how many images were dropped
        n_bad = unfiltered_df.height - self.records.height
        if n_bad:
            logger.warning("Dropped %d corrupt or non-image files", n_bad, extra={"markup": True})

        # WARN: debug only
        with pl.Config(fmt_str_lengths=50):  # make it a little longer for path
            pprint(self.records.sample(10, shuffle=True))

        # build class bins
        min_z: float = self.records.select(pl.min("z_value")).item()
        max_z: float = self.records.select(pl.max("z_value")).item()

        # Ensure max_z is included in the bins
        bin_edges: npt.NDArray[np.float64] = np.arange(min_z, max_z + self.class_steps, self.class_steps, float)
        self.num_bins: int = len(bin_edges) - 1
        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # keep for use in plotting function

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[ImageType, int]:
        """Retrieve an item from the dataset by index and return the image along with the object.

        Args:
            self: The object associated with the item.
            idx (int): The index of the item.

        Returns:
            Tuple[Image, int]: A tuple containing the data of the image, and its corresponding index in the total
                               dataset.

        """
        # grabs row at index, each column value in this row corresponds to its row via the dict structure
        record_row: dict[str, Any] = self.records.row(idx, named=True)
        relative_path: str = record_row["path"].item()  # Extract string path value

        absolute_csv_path = Path(self.metadata_csv_path_str) / relative_path  # Construct absolute path to each image

        # load hologram image with PIL
        try:
            # Translates pixels through given palette, "L" grayscale, "RGB" color
            img_pil = Image.open(absolute_csv_path).convert("L")
        except Exception as e:
            pprint(f"Error loading image {absolute_csv_path}: {e}")
            raise  # raise any PIL errors

        z_value: float = record_row["z_value"].item()  # generate class label from z_value

        # np.digitize returns indices of the bins to which each value of z belongs
        # it starts from 1 thus -1 for 0-based index
        cls_dig = np.digitize(z_value, self.bin_edges) - 1
        # assign these values
        # clip to ensure label is 0 <= label <= (number of bins - 1)
        cls = int(np.clip(cls_dig, 0, self.num_bins - 1))  # Ensure integer type

        # return the image, and a new instance of the class
        return img_pil, cls
