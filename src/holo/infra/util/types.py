# pyright: basic
from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, TypeVar

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


def check_dtype(type_check: str, expected_type, **kwargs) -> bool:
    """Check all inputs have same types/datatypse, else return exception."""
    # get first value to be checked against
    first_value: Any = next(iter(kwargs.values()))
    match type_check:
        # case for tensor items
        case "dtype":
            # if expected type is passed, check it against each item
            if expected_type is not None:
                all_correct_type = all(v.dtype == expected_type for v in kwargs.values())
            else:
                # otherwise ensure consistency with first dtype
                all_correct_type = all(v.dtype == first_value.dtype for v in kwargs.values())
            # not all the correct type, show each items dtype
            if all_correct_type is False:
                logger.error([(k, v.dtype) for k, v in kwargs.items()])
                return False
            return True

        case _:
            raise Exception(f"Unexpected type check argument {type_check}")
