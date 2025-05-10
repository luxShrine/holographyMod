from typing import Any

import numpy.typing as npt

from holo.util.log import logger


def print_list(input_list: list[Any]) -> None:
    """Small helper to print lists nicely."""
    print(*input_list, sep=",")


def convert_array_tolist_type(data_array: npt.NDArray[Any], data_type: Any):
    """Convert a numpy array to a list of a certain type."""
    data_list: list[Any] = []
    for i in range(len(data_array)):
        try:
            data_list.append(data_type(data_array[i]))
        except ValueError as ve:
            # Handle the exception by raising an error
            logger.exception(f"Cannot convert '{data_array[i]}' to {data_type.__name__}")
            raise ve
        except IndexError as ie:
            logger.exception(f"Cannot find index at '{i}' max index is {len(data_array)}")
            raise ie
    return data_list
