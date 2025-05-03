from typing import Any

import numpy as np
import numpy.typing as npt


def norm(data: npt.NDArray[Any]):
    max = np.max(data, keepdims=True)
    min = np.min(data, keepdims=True)
    normed_array = (data - min) / (max - min)
    return normed_array
