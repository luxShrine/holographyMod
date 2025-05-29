"""Exposes common paths useful for manipulating datasets and generating figures.

Styled after https://github.com/showyourwork/showyourwork/tree/main.

"""

from pathlib import Path


# ruff: noqa: D103
# Absolute path to the top level of the repository
# root = Path(__file__).resolve().parents[2].absolute()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4].absolute()


# Absolute path to the `src` folder
def src_root() -> Path:
    return repo_root() / "src"


# Absolute path to the `src/data` folder (contains datasets)
def data_root() -> Path:
    return src_root() / "data"


# Absolute path to the bridge data
def bridge100k_data() -> Path:
    return data_root() / "bridge100k"


# Absolute path to the Brownian_motion data
def brownian_data() -> Path:
    return data_root() / "Brownian_Motion_Strouhal_Analysis_Data"


# Absolute path to the primary train data
def MW_data() -> Path:
    return data_root() / "MW-Dataset"


# Absolute path to the norm/thal cells data
def NTcells_data() -> Path:
    return data_root() / "normal_and_thalassemic_cells"


# Absolute path to the phase only hologram data
def phase_data() -> Path:
    return data_root() / "Phase_Only_Holograms"


# Absolute path to the maynooth data
def maynooth_data() -> Path:
    return data_root() / "DHM_1"


# Absolute path to the DHM data
def DHM_1_data() -> Path:
    return data_root() / "DHM_1"


# Absolute path to the Keratinocyte HaCaT
def keratinocyte_data() -> Path:
    return DHM_1_data() / "DHM" / "Timelapse" / "Keratinocyte cell line HaCaT"


# Absolute path to the `src/static` folder (contains static images)
def static_root() -> Path:
    return src_root() / "static"


# Absolute path to the `src/holo` folder (contains holo package)
def holo_root() -> Path:
    return src_root() / "holo"


# Absolute path to the `src/tex` folder (contains the paper)
def tex_root() -> Path:
    return src_root() / "tex"


# Absolute path to the `src/tex/figures` folder (contains figure output)
def figures_tex() -> Path:
    return tex_root() / "figures"


# Absolute path to the `src/tex/output` folder (contains other user-defined output)
def output_tex() -> Path:
    return tex_root() / "output"
