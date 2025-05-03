import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt


@dataclass
class plotPred:
    """Class for storing plotting information."""

    y_test_pred: npt.NDArray[Any]
    y_test: npt.NDArray[Any]
    y_train_pred: npt.NDArray[Any]
    y_train: npt.NDArray[Any]
    yerr_train: npt.NDArray[Any]
    yerr_test: npt.NDArray[Any]
    title: str
    fname: Path
    save_fig: bool = True


def load_obj() -> list[plotPred]:
    """Load object of class plotPred."""
    try:
        with open("cats.json") as fd:
            return [plotPred(**x) for x in json.load(fd)]
    except FileNotFoundError:
        return []


def save_obj(c: plotPred):
    """Save object of class plotPred."""
    data = [asdict(x) for x in load_obj() + [c]]
    with open("train_data.json", "w") as fd:
        json.dump(data, fd)
