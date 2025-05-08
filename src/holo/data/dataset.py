import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from rich.pretty import pprint
from torch.utils.data import Dataset

import holo.util.paths as paths
from holo.util.crop import crop_max_square
from holo.util.log import logger
from holo.util.metadata import correct_data_csv

HOLO_DEF = paths.MW_data()  # prevents call in class

# TODO: create helper like paths for keeping units consistent


class HologramFocusDataset(Dataset[tuple[ImageType, int]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self,
        mode: str = "reg",
        hologram_dir: Path = HOLO_DEF,
        metadata_csv: str = "ODP-DLHM-Database.csv",
        crop_size: int = 512,
        class_steps: float = 5,
    ) -> None:
        """Assign properties to hologram dataset.

        Args:
        mode (str): Classification or regression type anaylisis for autofocus model.
        hologram_dir (Path): Base directory used to find relevant paths.
        metadata_csv (str): Path to CSV that maps images to properties, z distance and wavelength.
        crop_size (int): What size to crop image to.
        class_steps (float): How many bins to make in process of making discrete z values.

        """
        # my hologram dataset is a subclass of the torch.utils.data Dataset, this allows initializing both classes
        super().__init__()
        # assign initialized items
        self.mode: str = mode
        self.hologram_dir: Path = hologram_dir  # Store path for hologram directory
        self.metadata_csv_path_str: str = metadata_csv  # Store metadata csv name
        self.crop_size: int = crop_size  # assign the crop size
        # self.class_steps: float = class_steps * 1e6  # convert um -> m per bin

        path_to_csv: Path = self.hologram_dir / self.metadata_csv_path_str
        # cleanup data remove bad images, bad paths, records is where all the csv data is
        self.records: pl.DataFrame = correct_data_csv(path_to_csv, self.hologram_dir)

        # Now build all attributes from the filtered DataFrame:
        self.files = self.records["path"].to_list()
        self.z = (self.records["z_value"].to_numpy() * 1e-3).astype(np.float32)  # mm -> m
        self.wavelength = (self.records["Wavelength"].to_numpy() * 1e-6).astype(np.float32)  # um -> m

        # build bins
        z_mm = self.z * 1e3
        # get unique positions in mm
        uniq = np.unique(z_mm)
        # compute min non-zero step (in mm)
        diffs = np.diff(uniq)
        step_mm = float(np.min(diffs[diffs > 0]))
        step_m = step_mm * 1e-3
        step_m_mod = step_m * 100  # increase to make training coarser
        self.class_steps: float = step_m_mod  # convert to m

        # build bin edges at exactly each z-position
        self.bin_edges = np.arange(self.z.min(), self.z.max() + step_m_mod, step_m_mod, dtype=np.float64)

        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.num_bins = len(self.bin_centers)

        self.px = np.full_like(self.z, 3e-6, dtype=np.float32)  # NOTE: constant pixel size

        # check that the bins are reasonable
        if logger.isEnabledFor(logging.DEBUG):
            min_z: float = self.z.min()
            max_z: float = self.z.max()
            logger.debug(
                f"min z: {min_z}, max z: {max_z}, bin edges {self.bin_edges}, \
                edges length: {len(self.bin_edges)} current file: {__file__}"
            )
        # # Ensure max_z is included in the bins
        # self.bin_edges: npt.NDArray[np.float64] = np.arange(min_z, max_z + self.class_steps, self.class_steps, float)
        # bin_edges = np.arange(min_z, max_z + self.class_steps, self.class_steps, float)
        #
        #
        # self.num_bins: int = len(self.bin_edges) - 1
        # self.bin_edges = self.bin_edges
        # self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])  # keep for use in plotting function

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[ImageType, int]:
        """Retrieve an item from the dataset by index and return the image along with the object.

        Args:
            self: The object associated with the item.
            idx (int): The index of the item.

        Returns:
            Tuple[Image, int]: A tuple containing the data of the image, and it's corresponding bin index.

        """
        # grabs row at index, each column value in this row corresponds to its row via the dict structure
        record_row: dict[str, Any] = self.records.row(idx, named=True)
        relative_path = record_row["path"]  # Extract string path value

        # NOTE: This assumes that the CSV file lists the holograms as relative to itself
        absolute_csv_path = Path(self.metadata_csv_path_str) / relative_path

        # load hologram image with PIL
        try:
            # Translates pixels through given palette, "L" grayscale, "RGB" color
            img_pil = Image.open(absolute_csv_path).convert("RGB")
        except Exception as e:
            pprint(f"Error loading image {absolute_csv_path}: {e}")
            raise  # raise any PIL errors

        # if regression, use a continuous z value
        if self.mode == "reg":
            z_val = self.z[idx]
            return img_pil, z_val
        else:
            # if classification, we need the object
            z_value: float = record_row["z_value"]  # generate class label from z_value
            # np.digitize returns indices of the bins to which each value of z belongs
            # it starts from 1 thus -1 for 0-based index
            cls_dig = np.digitize(z_value, self.bin_edges) - 1
            # assign these values
            # clip to ensure label is 0 <= label <= (number of bins - 1)
            cls = int(np.clip(cls_dig, 0, self.num_bins - 1))  # Ensure integer type

            # return the image, and a new instance of the class
            return img_pil, cls


class HQDLHMDataset(HologramFocusDataset):
    """Loads HQ-DLHM-OD holograms.

    Returns: (tensor[1,H,W], float‑z [m], float‑λ [m], float‑px [m]).
    """

    def __init__(self, metadata_csv: str, crop: int = 224, mode: str = "reg"):
        super().__init__(metadata_csv=metadata_csv, crop_size=crop)  # inherit
        # edges = np.arange(self.z.min(), self.z.max() + self.class_steps, self.class_steps)
        bin_ids = np.digitize(self.z, self.bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, self.num_bins - 1)
        self.bin_ids = torch.tensor(bin_ids, dtype=torch.long)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])  # keep for use in plotting function
        self.mode = mode

        self.crop = crop

    def __getitem__(self, idx: int) -> tuple[ImageType, float, float, float]:  # type: ignore[override]
        """Overide default dataset indexing to retrieve the image, and its relevant attributes.

        Returns:
           (image_tensor[1,H,W], z [m], wavelength [m], pixel_size [m]).

        """
        # 1) load hologram as PIL.Image
        record_row: dict[str, Any] = self.records.row(idx, named=True)
        relative_path = record_row["path"]  # Extract string path value

        # NOTE: This assumes that the CSV file lists the holograms as relative to itself
        img_path = self.hologram_dir / relative_path
        img = Image.open(img_path).convert("L")
        # 2) optional center-crop via your util
        if self.crop:
            img = crop_max_square(img)
        # 3) return the PIL image, not a tensor
        return img, self.z[idx], self.wavelength[idx], self.px[idx]
