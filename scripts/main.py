#!/usr/bin/env -S uv run --script
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from PIL.Image import Image as ImageType

import holo.analysis.metrics as met
import holo.optics.reconstruction as rec
import holo.util.paths as paths
import holo.util.saveLoad as sl
from holo.train.auto_caller import train_autofocus_refactored
from holo.train.auto_classes import AutoConfig
from holo.util.crop import crop_max_square
from holo.util.log import logger
from holo.util.metadata import build_metadata_csv
from holo.util.normalize import norm

app = typer.Typer()


@app.command()
def plot_train():
    """Plot the data saved from autofocus training."""
    plot_info_list = sl.load_obj()
    plot_info = plot_info_list[0]  # unpack list of obj
    met.plot_actual_versus_predicted(
        np.array(plot_info.y_test_pred),
        np.array(plot_info.y_train_pred),
        np.array(plot_info.y_test),
        np.array(plot_info.y_train),
        np.array(plot_info.yerr_train),
        np.array(plot_info.yerr_test),
        plot_info.title,
        # plot_info.save_fig,
        True,
        str(plot_info.fname),
    )


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
        logger.warning(f"Path {ds_root} does not exist, defaulting to {paths.MW_data()}")
        ds_root = paths.MW_data()

    autofocus_config = AutoConfig(
        out_dir=out,
        meta_csv_name=metadata,
        backbone=backbone,
        batch_size=batch,
        epoch_count=ep,
        crop_size=crop,
        opt_lr=learn_rate,
        val_split=value_split,
        device_user=device_type,
    )

    _, _, plot_info = train_autofocus_refactored(autofocus_config)

    sl.save_obj(plot_info)


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
    z: float = 20e-3,
    dx: float = 1e-6,
):
    """Peform reconstruction on an hologram.

    Args:
        img_file_path: Path to image for reconstruction
        model_path: Path to trained model to use for torch optics anaylsis
        backbone: Model type being loaded
        crop_size: Pixel width and height of image
        wavelength: Wavelength of light used to capture the image (m)
        z: Distance of measurement (m)
        dx: Size of image px (m)

    """
    ckpt_file = Path("checkpoints") / model_path
    if not ckpt_file.exists():
        typer.secho(f" checkpoint not found at {ckpt_file}", fg="red")
        raise typer.Exit(1)
    pil_image: ImageType = Image.open(img_file_path).convert("RGB")
    recon, amp, phase = rec.torch_recon(img_file_path, wavelength, ckpt_file, crop_size, z, backbone, dx)  # type: ignore

    # normalize both images for comparison
    holo_org = np.array(crop_max_square(pil_image))  # crop, and convert to array
    n_org = norm(holo_org)
    n_recon = norm(recon)
    nrmsd, psnr = met.error_metric(n_org, n_recon, 255)
    met.plot_amp_phase(amp, phase)

    # error
    # train_nrmsd, train_psnr = error_metric(train_tgts, train_preds, max_px)
    # val_nrmsd, val_psnr = error_metric(val_tgts, val_preds, max_px)
    # print(f"[metrics]  Train  NRMSD={train_nrmsd:.4f} | PSNR={train_psnr:.2f} dB")
    # print(f"[metrics]  Val    NRMSD={val_nrmsd:.4f} | PSNR={val_psnr:.2f} dB")
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
