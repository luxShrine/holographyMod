# pyright: basic
from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, TypeVar

import numpy as np
import numpy.typing as npt
import pint
from beartype.vale import Is

_T_co = TypeVar("_T_co", covariant=True)

logger = logging.getLogger(__name__)   


# -- SINGLETON UNIT REGISTRY ---------------------------------------------------------------------
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity  # short alias

u = ureg  # short namespace, for u.um, u.nm, u.m


# -- NUMPY HELPERS -------------------------------------------------------------------------------
type Np1Array64 = npt.NDArray[np.float64]
type Np1Array32 = npt.NDArray[np.float32]

# -- RANGE GUARDS FROM BEARTYPE ------------------------------------------------------------------
type Nanometers = Annotated[Q_, Is[lambda q: 200 * u.nm <= q <= 2000 * u.nm]]
type Micrometers = Annotated[Q_, Is[lambda q: -1_000 * u.um <= q <= 1_000 * u.um]]


class AnalysisType(Enum):
    """Restrict analysis variable to known strings."""

    CLASS = "class"
    REG = "reg"


class DisplayType(Enum):
    """Restrict display variable to known strings."""

    SHOW = "show"
    SAVE = "save"
    BOTH = "both"


class UserDevice(Enum):
    """Restrict device variable to known device strings."""

    CPU = "cpu"
    CUDA = "cuda"
