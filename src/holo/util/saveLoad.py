import json
from dataclasses import asdict
from dataclasses import dataclass

# from pathlib import Path
from typing import Any

from holo.util.log import logger

# import numpy.typing as npt


@dataclass
class plotPred:
    """Class for storing plotting information."""

    # z_test_pred: npt.NDArray[Any]
    # z_test: npt.NDArray[Any]
    # z_train_pred: npt.NDArray[Any]
    # z_train: npt.NDArray[Any]
    # zerr_train: npt.NDArray[Any]
    # zerr_test: npt.NDArray[Any]
    z_test_pred: list[Any]
    z_test: list[Any]
    z_train_pred: list[Any]
    z_train: list[Any]
    zerr_train: list[Any]
    zerr_test: list[Any]
    title: str
    fname: str
    save_fig: bool = True


# TODO: currently a static name <05-09-25>
def load_obj() -> list[plotPred]:
    """Load saved json containing object of class plotPred."""
    try:
        with open("train_data.json") as fd:
            return [plotPred(**x) for x in json.load(fd)]
    except FileNotFoundError:
        logger.exception(FileNotFoundError)
        return []


def save_obj(c: plotPred):
    """Save object of class plotPred."""
    data = [asdict(x) for x in load_obj() + [c]]
    with open("train_data.json", "w") as fd:
        json.dump(data, fd)
