from __future__ import annotations

from enum import Enum
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pint
from beartype.vale import Is

from holo.data.transformed_dataset import TransformedDataset

###############################################################################
# SINGLETON UNIT REGISTRY
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity  # short alias

u = ureg  # short namespace, for u.um, u.nm, u.m

###############################################################################
# NUMPY HELPERS
type Np1Array64 = npt.NDArray[np.float64]
type Np1Array32 = npt.NDArray[np.float32]

###############################################################################
# RANGE GUARDS FROM BEARTYPE
# Any nanometer qantity must lie in [200, 2000] nm
Nanometers = Annotated[Q_, Is[lambda q: 200 * u.nm <= q <= 2000 * u.nm]]
# Any micro quantity must lie in [-1000, 1000] um
Micrometers = Annotated[Q_, Is[lambda q: -1_000 * u.um <= q <= 1_000 * u.um]]


class DisplayType(Enum):
    SHOW = "show"
    SAVE = "save"
    BOTH = "both"


class UserDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"


################################################################################
# EPOCH CHECKS
def check_dataload(
    selected_train_loader,
    selected_val_loader,
    evaluation_metric,
    train_ds: TransformedDataset,
    val_ds: TransformedDataset,
    z_sig,
    z_mu,
):
    if selected_train_loader and selected_val_loader:
        # TODO: what do i expect, how to test for it? apply to all sub parts <luxShrine >
        if evaluation_metric:
            if train_ds and val_ds:
                if z_sig and z_mu:
                    print("")
