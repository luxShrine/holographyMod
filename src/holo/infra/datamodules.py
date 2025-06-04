from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

import holo.infra.util.paths as paths
from holo.infra.util.image_processing import correct_data_csv
from holo.infra.util.types import AnalysisType, Np1Array32, Np1Array64

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)
HOLO_DEF = paths.MW_data()


# TODO: transform section not needed
class HologramFocusDataset(Dataset[tuple[ImageType, np.float64]]):
    """Store dataset information relevant to reconstruction."""

    def __init__(
        self,
        mode: AnalysisType,
        num_classes: int | None,
        csv_file_strpath: str = (HOLO_DEF / Path("ODP-DLHM-Database.csv")).as_posix(),
    ) -> None:
        # inherit from torch dataset
        # super().__init__()
        # Create set of records to draw from
        csv_file_path: Path = Path(csv_file_strpath)
        self.records: pl.DataFrame = correct_data_csv(csv_file_path.resolve(), HOLO_DEF.resolve())
        self.mode: AnalysisType = mode
        self.num_classes: int | None = num_classes

        self.holo_dir: Path = csv_file_path.parent
        self.metadata_csv_path_str: str = csv_file_strpath
        self.paths: list[Path] = [self.holo_dir / Path(p) for p in self.records["path"].to_list()]

        # Store magnitudes in meters
        self.z_m: Np1Array64 = self.records["z_value"].to_numpy()
        self.wavelength_m: Np1Array64 = self.records["Wavelength"].to_numpy()
        # NOTE: constant pixel size
        self.px_m: Np1Array32 = np.full(len(self.z_m), 3.8e-6, dtype=np.float32)

        # the z depth on its own cannot be passed to the model, it must be
        # converted to a set of bins, as integers. to do so digitize array => bins
        z_uniq = np.unique(self.z_m)
        logger.debug(
            f"unique depths sample (m): {z_uniq[:10]} \n total: {len(np.unique(self.z_m))}"
        )

        # TODO: make classification items dependent on AnalysisType
        if self.mode == AnalysisType.CLASS:
            if self.num_classes is None or self.num_classes <= 0:
                raise ValueError("num_classes must be a positive integer for classification.")
            # upper and lower bound
            min_z: np.float64
            max_z: np.float64
            min_z, max_z = self.z_m.min(), self.z_m.max()
            # create as many bins as selected number of classes, thus num_classes + 1 edges
            self.bin_edges: Np1Array64 = np.linspace(
                min_z, max_z, self.num_classes + 1, dtype=np.float64
            )

            # np.digitize returns 1-based indices for bins normally.
            # bins[i-1] <= x < bins[i] -> i.
            # x < bins[0] -> 0. x >= bins[num_classes] (last edge) -> num_classes + 1.
            raw_binned_indices: npt.NDArray[np.intp] = np.digitize(self.z_m, self.bin_edges)

            # Clip to ensure indices are within [1, num_classes] then subtract 1 for 0-indexing
            # This maps values < min_z to bin 0, and values >= max_z to bin num_classes-1
            self.z_bins: npt.NDArray[np.intp] = np.clip(raw_binned_indices, 1, self.num_classes) - 1

            self.bin_centers: Np1Array64 = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            logger.info(
                f"Created {self.num_classes} bins for classification with "
                + f"centers: {self.bin_centers}"
            )

        # TODO: Implement automatic bin creation logic in some form
        # bins ought to wide enough to be populated, but not too wide as to loose meaning
        # first, establish range of bins, limited as to not become negative
        # potential_lb = self.z_m.min() - self.z_m.std()
        # potential_lb = self.z_m.min()

        # lower_bound = potential_lb if potential_lb >= 0 else self.z_m.min()
        # upper_bound = self.z_m.max()
        # # using the Freedman–Diaconis rule for bin size, first IQR
        # interquartile_range = np.percentile(self.z_m, 75) - np.percentile(self.z_m, 25)
        # # bin width is thus 2 * (IQR * (N)^-3/2)

        # bin_width = (2 * (interquartile_range / (len(z_uniq) ** (3 / 2)))) * 100
        # bins = np.arange(lower_bound, upper_bound, bin_width)
        # logger.info(f"Length of bins is: {len(bins)}")
        #
        # self.z_bins: npt.NDArray[np.int32] = np.digitize(self.z_m, bins)

    def __len__(self) -> int:
        """Return the lengh of entries in dataframe."""
        return len(self.records)

    @override
    def __getitem__(self, idx: int) -> tuple[ImageType, np.float64]:
        """Return image and z depth value."""
        # ensure image is read properly
        try:
            img: ImageType = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            logger.exception(f"Error loading image {self.paths[idx]}: {e}")
            raise
        # Continuous if regression, bins if classification
        label = self.z_m[idx] if self.mode == AnalysisType.REG else self.z_bins[idx]

        # Return quantities for clarity
        return (img, label)
