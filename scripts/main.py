#!/usr/bin/env -S uv run --script
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from PIL.Image import Image as ImageType

import holo.io.paths as paths
from holo.io.metadata import build_metadata_csv
from holo.optics.reconstruction import fresnel_numpy
from holo.train.autofocus import train_autofocus
from holo.util.crop import crop_max_square

app = typer.Typer()


@app.command()
def main_train(
    ds_root: Path,
    metadata: str = "ODP-DLHM-Database.csv",
    out: str = "checkpoints",
    backbone: str = "efficientnet_b4",
    crop: int = 512,
    value_split: float = 0.2,
    batch: int = 16,
    ep: int = 10,
    learn_rate: float = 1e-4,
    device_type: str = "cuda",
) -> None:
    """Train the autofocus model based on supplied dataset.

    Args:
        ds_root:  Directory containing hologram images.
        metadata: Path to the metadata CSV file.
        out: Directory to save checkpoints and logs.
        backbone: Model backbone name.
        batch: Training batch size.
        crop: Size to crop images to.
        ep: Number of training epochs.
        learn_rate: How fast should the model change epoch to epoch
        value_split: Fraction of data for validation.
        device_type: Device ('cuda' or 'cpu').

    """
    # NOTE: quick test settings
    # batch = 8
    # crop = 256
    # ep = 2
    # learn_rate = 1e-3

    # NOTE: substantial test settings
    # crop = 512
    # value_split = 0.2
    # batch = 16
    # ep = 50
    # learn_rate = 1e-4

    if ds_root.exists():
        pass
    else:
        ds_root = paths.MW_data()

    _ = train_autofocus(
        hologram_dir=ds_root,
        metadata_csv=metadata,
        out_dir=out,
        backbone=backbone,
        batch_size=batch,
        epochs=ep,
        crop_size=crop,
        lr=learn_rate,
        val_split=value_split,
        device=device_type,
    )


# TODO: decide/find proper units for this
@app.command()
def main_recon(img_file_path: str, wavelength: float = 530e-9, z: float = 300e-6, dx: float = 1.12e-6):
    """Peform reconstruction on an image.

    Args:
        img_file_path: Path to image for reconstruction
        wavelength: wavelength of light used to capture the image (nm)
        z: distance of measurement in terms of height (um)
        dx: size of image px (um)

    """
    pil_image: ImageType = Image.open(img_file_path)

    cropped_image = np.asarray(crop_max_square(pil_image))

    fresnel_numpy(cropped_image, dx, wavelength, z)


@app.command()
def create_meta(hologram_directory: str, out_directory: str):
    """Build csv containing metadata of images in hologram directory."""
    hologram_dir = Path(hologram_directory)
    out_dir = Path(out_directory)

    build_metadata_csv(
        hologram_dir,
        out_dir,
    )


if __name__ == "__main__":
    app()
