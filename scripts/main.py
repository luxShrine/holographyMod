#!/usr/bin/env -S uv run --script
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from PIL.Image import Image as ImageType

import holo.io.paths as paths
import holo.optics.reconstruction as rec
from holo.analysis.metrics import error_metric
from holo.analysis.metrics import plot_amp_phase
from holo.io.metadata import build_metadata_csv
from holo.train.autofocus import train_autofocus
from holo.util.normalize import norm

app = typer.Typer()


@app.command()
def train(
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
    # ep = 3
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


# NOTE: main database used ODP-DLHM-Database specs:
# 4640x3506 [px]
# 8bit-monochromatic
# 3.8 [um] pixel size
# 405, 510, 654, [nm] laser
# 0.5-4 [um]
@app.command()
def reconstruction(
    img_file_path: str,
    model_path: str = "best_model.pth",
    backbone: str = "efficientnet_b4",
    crop_size: int = 512,
    wavelength: float = 530e-9,
    z: float = 300e-6,
    dx: float = 1e-6,
):
    """Peform reconstruction on an hologram.

    Args:
        img_file_path: Path to image for reconstruction
        model_path: Path to trained model to use for torch optics anaylsis
        backbone: Model type being loaded
        crop_size: Pixel width and height of image
        wavelength: Wavelength of light used to capture the image (nm)
        z: Distance of measurement (um)
        dx: Size of image px (um)

    """
    ckpt_file = Path("checkpoints") / model_path
    if not ckpt_file.exists():
        typer.secho(f" checkpoint not found at {ckpt_file}", fg="red")
        raise typer.Exit(1)
    pil_image: ImageType = Image.open(img_file_path).convert("RGB")
    recon, amp, phase = rec.torch_recon(img_file_path, wavelength, ckpt_file, crop_size, z, backbone, dx)  # type: ignore

    # normalize both images for comparison
    holo_org = np.array(pil_image)
    n_org = norm(holo_org)
    n_recon = norm(recon)
    nrmsd, psnr = error_metric(n_org, n_recon, 255)
    plot_amp_phase(img_file_path, amp, phase, nrmsd=nrmsd, psnr=psnr)

    # error
    typer.echo(f"the psnr is {psnr} with nrmsd: {nrmsd}")


@app.command()
def create_meta(hologram_directory: str, out_directory: str):
    """Build csv containing metadata of images in hologram directory."""
    hologram_dir = Path(hologram_directory)
    out_dir = Path(out_directory)

    _ = build_metadata_csv(
        hologram_dir,
        out_dir,
    )


if __name__ == "__main__":
    app()
