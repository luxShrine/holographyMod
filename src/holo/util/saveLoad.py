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

    z_test_pred: list[Any]
    z_test: list[Any]
    z_train_pred: list[Any]
    z_train: list[Any]
    zerr_train: list[Any]
    zerr_test: list[Any]
    title: str  # TODO: shouldnt have static names, maybe parse this string for what is being measured to pass to plot
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
