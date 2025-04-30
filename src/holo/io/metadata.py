from pathlib import Path

import polars as pl


def parse_info(info_file: Path):
    """For each info.txt, extract data from it."""
    info: dict[str, float] = {}
    for line in info_file.read_text().splitlines():
        if not line.strip():
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().rstrip("um")  # strip units
        info[key] = float(val)
    return info


def build_metadata_csv(root: Path, out: Path) -> pl.DataFrame:
    """Return a dataframe containing the path to each image file with its data from the info.txt file.

    Args:
        root: directory of dataset
        out: directory to save resulting csv to
    returns:
        dataframe containing dataset paths linked to image's information

    """
    rows: list[dict[str, (str | float)]] = []
    for info in root.rglob("info.txt"):
        meta = parse_info(info)
        for img in info.parent.rglob("*.[jp][np]g"):
            rows.append(
                {
                    "path": str(img.relative_to(root)),
                    **meta,  # unpacks a dictionary into keyword arguments
                }
            )
    df = pl.DataFrame(rows)
    df.glimpse()

    # if output is a directory, save to that directory
    if Path.is_dir(out):
        dir_out = out / "metadata.csv"
        df.write_csv(dir_out, separator=";")
    else:
        df.write_csv(out, separator=";")  # otherwise, save to specified file

    return df
