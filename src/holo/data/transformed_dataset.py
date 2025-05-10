from collections.abc import Callable
from typing import Any
from typing import TypeVar

from PIL.Image import Image as ImageType
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import Subset

# generic type variables needed
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


class TransformedDataset(Dataset[tuple[Any, Any]]):
    """Apply transforms by creating a wrapper dataset to avoid modifying original HologramFocusDataset."""

    def __init__(self, subset: Subset[_T_co], transform: Callable[[Any], Any] | None, dataset_bin_ids: Tensor):
        """Initialize the TransformedDataset.

        Args:
            subset (Subset[_T_co]): The subset of the original dataset.
            transform (Optional[Callable[[Any], Any]]): The transformation function to apply.
            It takes the first image and returns the transformed element.

        """
        self.subset: Subset[_T_co] = subset
        self.transform = transform
        self.dataset_bin_ids = dataset_bin_ids

    def __getitem__(self, index: int) -> tuple[ImageType, int]:
        """Retrieve an item from the dataset by index and applies the transform.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[Any, Any]: A tuple containing the (potentially transformed) data
                             and its corresponding label/target. The exact types
                             depend on the subset and the transform.

        """
        original_idx = self.subset.indices[index]  # type: ignore
        item = self.subset[index]  # type: ignore
        x = item[0]  # type: ignore
        if self.transform:
            # Transform applies to x
            x = self.transform(x)  # Type of x might change here
        return x, self.dataset_bin_ids[original_idx]  # type: ignore

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return len(self.subset)  # type: ignore

    def __getattr__(self, name: str):
        """Forward attribute look-ups to the underlying dataset, so items can be extracted."""
        try:
            return getattr(self.subset.dataset, name)  # type: ignore
        except AttributeError as e:
            raise AttributeError(name) from e
