"""Exposes common paths useful for manipulating datasets and generating figures.

Styled after https://github.com/showyourwork/showyourwork/tree/main.

"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def path_check(**kwargs: Path) -> None:
    """Ensure all paths passed in exist."""
    for variable, path_to_check in kwargs.items():
        try:
            assert isinstance(path_to_check, Path), f"{path_to_check} is not a path."
            valid_path: bool = path_to_check.exists()
            if not valid_path:
                raise Exception(f"Path for {variable} does not exist at {path_to_check}")
        except Exception as e:
            logger.error(f"{variable} not processed as path: {path_to_check}")
            raise e


def repo_root() -> Path:
    """Return the path to the repository root."""
    return Path(__file__).resolve().parents[4].absolute()


def src_root() -> Path:
    """Return the path to the ``src`` directory."""
    return repo_root() / "src"


def data_root() -> Path:
    """Return the directory that holds datasets (src/data)."""
    return src_root() / "data"


def bridge100k_data() -> Path:
    """Path to the ``bridge100k`` dataset (src/data/bridge100k)."""
    return data_root() / "bridge100k"


def brownian_data() -> Path:
    """Path to the Brownian motion dataset."""
    return data_root() / "Brownian_Motion_Strouhal_Analysis_Data"


def MW_data() -> Path:
    """Path to the primary training dataset (src/data/MW-Dataset)."""
    return data_root() / "MW-Dataset"


def NTcells_data() -> Path:
    """Path to the normal and thalassemic cells dataset."""
    return data_root() / "normal_and_thalassemic_cells"


def phase_data() -> Path:
    """Path to the phase-only hologram dataset."""
    return data_root() / "Phase_Only_Holograms"


def maynooth_data() -> Path:
    """Path to the Maynooth dataset."""
    return data_root() / "DHM_1"


def DHM_1_data() -> Path:
    """Path to the DHM dataset."""
    return data_root() / "DHM_1"


def keratinocyte_data() -> Path:
    """Path to the Keratinocyte HaCaT timelapse dataset."""
    return DHM_1_data() / "DHM" / "Timelapse" / "Keratinocyte cell line HaCaT"


def static_root() -> Path:
    """Return path to the directory storing static images (src/static)."""
    return src_root() / "static"


def holo_root() -> Path:
    """Return path to the holo package root (src/holo)."""
    return src_root() / "holo"


def tex_root() -> Path:
    """Return path to the ``tex`` directory (src/tex)."""
    return src_root() / "tex"


def figures_tex() -> Path:
    """Path where LaTeX figures are written (src/tex/figures)."""
    return tex_root() / "figures"


def output_tex() -> Path:
    """Path for miscellaneous LaTeX output files (src/tex/output)."""
    return tex_root() / "output"
