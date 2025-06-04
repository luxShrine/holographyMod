import logging
from random import randint

from holo.infra.datamodules import HologramFocusDataset
from holo.infra.util.types import Q_, u

logger = logging.getLogger(__name__)


def check_units(vars_to_check: dict[Q_, u]) -> bool:
    i = 0
    for v, units in vars_to_check.items():
        if v.u == units:
            continue
        i += 1
        logger.error(f"found unexpected units, expected {units} but got {v.to_compact}")
    return i != 1


def test_base(ds: HologramFocusDataset) -> None:
    # -- Broad Tests -----------------------------------------------------------------------------
    if not isinstance(ds, HologramFocusDataset):
        raise RuntimeError("dataset is not HologramFocusDataset")
    if len(ds) < 1:
        raise RuntimeError("dataset length is zero.")
    if len(ds) < 10:
        raise RuntimeError("dataset contains less than 50 images, too few to train on.")
    length_ds = len(ds)
    logger.info(f"length of base Hologram dataset is {length_ds}")
    # -- Test Types in __get_item__ --------------------------------------------------------------
    _image, _label = ds[randint(0, length_ds - 1)]
    # z_m, wavelength_m, pixel_size_m = label

    # if not isImageType(image) or not isinstance(wavelength_m, Q_) or not isinstance(z_m, Q_):
    #
    #     print(
    #         "==== Unexpected Types in Hologram ==== "
    #         f"image not imagetype, is {type(image)}"
    #         f"z not units, is {type(z_m)}"
    #         f"wavelength not units, is {type(wavelength_m)}"
    #         f"pixel_size_m not units, is {type(pixel_size_m)}"
    #     )

    # -- Test Units in __get_item__ --------------------------------------------------------------
    # create dict of quantities, units
    # quant_dict = {
    #     z_m: u.m,
    #     wavelength_m: u.m,
    #     pixel_size_m: u.m,
    # }
    # if not check_units(quant_dict):
    #     raise RuntimeError
