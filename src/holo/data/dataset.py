import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
        class_steps: float = 50,
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
        # ----------------------------------------------------------------------

        # assign initialized items
        self.mode: str = mode
        self.hologram_dir: Path = hologram_dir  # Store path for hologram directory
        self.metadata_csv_path_str: str = metadata_csv  # Store metadata csv name
        self.crop_size: int = crop_size  # assign the crop size

        # ----------------------------------------------------------------------

        # cleanup data remove bad images, bad paths, records is where all the csv data is
        path_to_csv: Path = self.hologram_dir / self.metadata_csv_path_str
        self.records: pl.DataFrame = correct_data_csv(path_to_csv, self.hologram_dir)

        # Now build all attributes from the filtered DataFrame:
        self.files = self.records["path"].to_list()
        z_raw = self.records["z_value"].to_numpy()
        wavelength_raw = self.records["Wavelength"].to_numpy()

        # convert once to (m) (float64 -> float32 for saving computing)
        self.z_m: npt.NDArray[np.float32] = (z_raw * 1e-6).astype("float32")
        self.wavelength_m: npt.NDArray[np.float32] = (wavelength_raw * 1e-6).astype("float32")

        # pixel size is constant in (m)
        self.px_m: npt.NDArray[np.float32] = np.full_like(self.z_m, 3e-6, dtype="float32")

        # ----------------------------------------------------------------------
        # normalize z value for regression

        self.z_mu: float = float(self.z_m.mean())
        self.z_sigma: float = float(self.z_m.std())
        self.z_norm: npt.NDArray[np.float32] = (self.z_m - self.z_mu) / self.z_sigma

        # ----------------------------------------------------------------------
        # build bins for classification

        self.bin_step_m = class_steps * 1e-6  # Convert input (um) to (m)

        min_z_m = self.z_m.min()
        max_z_m = self.z_m.max()

        if self.bin_step_m <= 0:
            logger.warning(
                f"Input class_steps ({class_steps} um) resulted in non-positive bin_step_meters. "
                f"Attempting to derive a step from data."
            )
            unique_z_m = np.unique(self.z_m)
            if len(unique_z_m) > 1:
                diffs_m = np.diff(unique_z_m)
                positive_diffs_m = diffs_m[diffs_m > 0]
                if len(positive_diffs_m) > 0:
                    self.bin_step_m = float(np.min(positive_diffs_m))
                else:
                    self.bin_step_m = 1e-6  # Default small step if only non-positive diffs
                    logger.warning("Could not determine positive data-derived step. Using default 1e-6 m.")
            else:  # Single unique z_value or empty
                self.bin_step_m = 1e-6  # Default small step
                logger.warning("Single unique z-value or no z-values. Using default step 1e-6 m.")

        # Create bin edges ensuring the max value is covered
        # Add a small fraction of step to max_z_m to ensure arange includes it if it's a boundary
        self.bin_edges_m = np.arange(min_z_m, max_z_m + self.bin_step_m / 2, self.bin_step_m, dtype=np.float32)

        if len(self.bin_edges_m) < 2:  # Not enough edges to form a bin
            # Create at least one bin covering the range, or a small default bin if min_z_m == max_z_m
            if min_z_m == max_z_m:
                self.bin_edges_m = np.array(
                    [min_z_m - self.bin_step_m / 2, max_z_m + self.bin_step_m / 2], dtype=np.float32
                )
            else:
                self.bin_edges_m = np.array(
                    [min_z_m, max_z_m + self.bin_step_m], dtype=np.float32
                )  # Ensure it's max_z_m + step
            logger.warning(f"Adjusted bin_edges_meters due to insufficient initial edges: {self.bin_edges_m}")

        self.bin_centers_m = 0.5 * (self.bin_edges_m[:-1] + self.bin_edges_m[1:])
        self.num_bins = len(self.bin_centers_m)

        if self.num_bins == 0 and len(self.records) > 0:
            logger.error(
                "Number of bins is 0 despite having records. \
            This indicates an issue with z-values or bin_step_meters."
            )
            # Fallback: create a single bin (should ideally not be reached if above logic is sound)
            self.bin_edges_m = np.array([min_z_m, max_z_m if max_z_m > min_z_m else min_z_m + self.bin_step_m])
            self.bin_centers_m = np.array([(self.bin_edges_m[0] + self.bin_edges_m[1]) / 2.0])
            self.num_bins = 1

        # check that the bins are reasonable
        if logger.isEnabledFor(logging.DEBUG):
            min_z_m: float = self.z_m.min()
            max_z_m: float = self.z_m.max()
            logger.debug(
                f"min z: {min_z_m}, max z: {max_z_m}, bin edges {self.bin_edges_m}, \
                edges length: {len(self.bin_edges_m)} current file: {__file__}"
            )

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
            z_val_m = self.z_m[idx]
            return img_pil, z_val_m
        else:
            # if classification, we need the object
            z_val_m = self.z_m[idx]  # generate class label from z_value
            # np.digitize returns indices of the bins to which each value of z belongs
            # it starts from 1 thus -1 for 0-based index
            cls_dig: npt.ArrayLike = np.digitize(z_val_m, self.bin_edges_m) - 1
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
        bin_ids_dig = np.digitize(self.z_m, self.bin_edges_m) - 1
        bin_ids = np.clip(bin_ids_dig, 0, self.num_bins - 1)
        self.bin_ids = torch.tensor(bin_ids, dtype=torch.long)
        self.bin_centers = self.bin_centers_m  # keep for use in plotting function
        self.mode = mode
        self.crop = crop

    def __getitem__(self, idx: int) -> tuple[ImageType, float, float, float]:  # type: ignore[override]
        """Overide default dataset indexing to retrieve the image, and its relevant attributes.

        Returns:
           (image_tensor[1,H,W], z [m], wavelength [m], pixel_size [m]).

        """
        # load hologram as PIL.Image
        record_row: dict[str, Any] = self.records.row(idx, named=True)
        relative_path = record_row["path"]  # Extract string path value

        # NOTE: This assumes that the CSV file lists the holograms as relative to itself
        img_path = self.hologram_dir / relative_path
        img = Image.open(img_path).convert("L")
        # center-crop via your util
        if self.crop:
            img = crop_max_square(img)
        # return the PIL image, not a tensor
        return img, self.z_m[idx], self.wavelength_m[idx], self.px_m[idx]
