from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

import holo.infra.util.paths as paths
from holo.infra.util.image_processing import correct_data_csv
from holo.infra.util.types import Q_, AnalysisType, Np1Array32, Np1Array64

if TYPE_CHECKING:
    import polars as pl
    from torch import Tensor


logger = logging.getLogger(__name__)
HOLO_DEF = paths.MW_data()


class HologramFocusDataset(Dataset[tuple[ImageType, Q_, Q_, Np1Array32]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self, mode: AnalysisType, csv_file: str, transform=None, target_transform=None
    ) -> None:
        # inherit from torch dataset
        # super().__init__()
        # Create set of records to draw from
        csv_file_path: Path = HOLO_DEF / Path(csv_file)
        self.records: pl.DataFrame = correct_data_csv(csv_file_path.resolve(), HOLO_DEF.resolve())

        self.holo_dir = csv_file_path.parent
        self.metadata_csv_path_str = csv_file
        self.paths = [self.holo_dir / Path(p) for p in self.records["path"].to_list()]

        # Store magnitudes in meters
        self.z_m: Np1Array64 = self.records["z_value"].to_numpy()
        self.wavelength_m: Np1Array64 = self.records["Wavelength"].to_numpy()
        # NOTE: constant pixel size
        self.px_m = np.full(len(self.z_m), 3.8e-6, dtype=np.float32)

        # Transforms & labels
        self.transform = transform
        self.target_transform = target_transform
        # the z depth on its own cannot be passed to the model, it must be
        # converted to a set of bins, as integers. to do so digitize array => bins

        z_uniq = np.unique(self.z_m)
        logger.debug(
            f"unique depths sample (m): {z_uniq[:10]} \n total: {len(np.unique(self.z_m))}"
        )

        # TODO: make classification items dependent on AnalysisType

        # TODO: make bins more robust

        # bins ought to wide enough to be populated, but not too wide as to loose meaning
        # first, establish range of bins, limited as to not become negative
        # potential_lb = self.z_m.min() - self.z_m.std()
        potential_lb = self.z_m.min()
        lower_bound = potential_lb if potential_lb >= 0 else self.z_m.min()
        upper_bound = self.z_m.max()
        # using the Freedman–Diaconis rule for bin size, first IQR
        interquartile_range = np.percentile(self.z_m, 75) - np.percentile(self.z_m, 25)
        # bin width is thus 2 * (IQR * (N)^-3/2)
        bin_width = (2 * (interquartile_range / (len(z_uniq) ** (3 / 2)))) * 100
        bins = np.arange(lower_bound, upper_bound, bin_width)
        logger.info(f"Length of bins is: {len(bins)}")

        self.z_bins: npt.NDArray[int] = np.digitize(self.z_m, bins)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> tuple[ImageType | Tensor, np.float64 | Tensor]:  # Return Quantities
        # ensure image is read properly
        try:
            img: ImageType = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            logger.exception(f"Error loading image {self.paths[idx]}: {e}")
            raise
        # create easy to grab "label", cannot have units
        label = self.z_bins[idx]

        # if transform is passed, provide the transformed image rather than raw
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        # Return quantities for clarity
        return (img, label)
