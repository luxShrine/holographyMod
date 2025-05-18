from pathlib import Path

#!/usr/bin/env -S uv run --script
import numpy as np
import typer
from PIL import Image
from PIL.Image import Image as ImageType

import holo.util.paths as paths
import holo.util.saveLoad as sl
from holo.util import log as log_setup
from holo.util.crop import crop_max_square
from holo.util.log import logger
from holo.util.log import set_verbosity
from holo.util.metadata import build_metadata_csv
from holo.util.normalize import norm
from holo.util.paths import static_root

app = typer.Typer(help="Hologram autofocus trainer / evaluator")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print DEBUG messages to the terminal"),
    log_file: Path = typer.Option("debug.log", "--log-file", help="Path to the log file"),
) -> None:
    """Set global options that apply to every sub-command."""
    # Send INFO vs DEBUG to the Rich console
    set_verbosity(verbose)

    # If the user moved the log file, swap the FileHandler target.
    if str(log_file) != log_setup.file_handler.baseFilename:
        log_setup.file_handler.close()
        log_setup.file_handler.baseFilename = str(log_file)
        log_setup.file_handler.stream = open(log_file, "a", encoding="utf-8")
        logger.info("Now logging to %s", log_file)


@app.command()
def train(
    ds_root: Path = typer.Argument(help="Directory containing hologram images."),
    metadata: str = typer.Option("ODP-DLHM-Database.csv", "--meta", help="Path to the metadata CSV file."),
    out: str = typer.Option("checkpoints", help="Directory to save checkpoints and logs."),
    backbone: str = typer.Option("efficientnet_b4", "--backbone", "--bb", help="Model backbone name."),
    crop: int = typer.Option(512, "--crop", "--c", help="Size to crop images to."),
    value_split: float = typer.Option(0.2, "--vs", help="Fraction of data for validation."),
    batch: int = typer.Option(16, "--batch", "--ba", help="Training batch size."),
    ep: int = typer.Option(10, help="Number of training epochs."),
    learn_rate: float = typer.Option(1e-4, "--lr", help="How fast should the model change epoch to epoch"),
    device_type: str = typer.Option("cuda", "--device", help="Device ('cuda' or 'cpu')."),
    analysis: bool = typer.Option(False, "--classfication", "-c", help="Change analysis type to classification"),
) -> None:
    """Train the autofocus model based on supplied dataset."""
    from holo.train.auto_caller import train_autofocus_refactored
    from holo.train.auto_classes import AutoConfig
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
    #
    mode: str = "reg"
    if analysis:
        mode = "class"
    else:
        pass

    # WARN: not robust for all types, will need a change <05-09-25>
    if backbone == "vit" and crop != 224:
        logger.error(f"backbone of type {backbone} requires a crop size of 224, defaulting to appropriate crop size")

    if ds_root.exists():
        actual_ds_root = ds_root
        pass
    else:
        actual_ds_root = paths.MW_data()
        logger.warning(f"Path {ds_root} does not exist, defaulting to {actual_ds_root}")

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
        auto_method=mode,
    )

    _, _, plot_info = train_autofocus_refactored(autofocus_config)

    sl.save_obj(plot_info)


@app.command()
def plot_train(
    analysis: bool = typer.Option(False, "--classfiication", "-c", help="Change analysis type to classification"),
    show: bool = typer.Option(True, "--s", help="Save the output plots, or display them."),
):
    """Plot the data saved from autofocus training."""
    import holo.analysis.plots as plots  # performance reasons, import locally in function

    plot_info_list = sl.load_obj()
    plot_info = plot_info_list[0]  # unpack list of obj
    if analysis:
        # TODO: add method for choosing plot based on regression or not, maybe save autofocus type? <05-09-25>
        met.plot_actual_versus_predicted(
            np.array(plot_info.z_test_pred),
            np.array(plot_info.z_test),
            np.array(plot_info.z_train_pred),
            np.array(plot_info.z_train),
            np.array(plot_info.zerr_train),
            np.array(plot_info.zerr_test),
            plot_info.title,
            show,
            str(plot_info.fname),
        )
    else:
        s_root = static_root().as_posix()
        plots.plot_residual_vs_true(
            np.array(plot_info.z_train),
            np.array(plot_info.z_train_pred),
            title="Residual vs True depth (train)",
            savepath=(s_root / Path("residual_vs_true_train.png")).as_posix(),
            show=show,
        )
        plots.plot_residual_vs_true(
            np.array(plot_info.z_test),
            np.array(plot_info.z_test_pred),
            title="Residual vs True depth (val)",
            savepath=(s_root / Path("residual_vs_true_val.png")).as_posix(),
            show=show,
        )

        plots.plot_violin_depth_bins(
            np.array(plot_info.z_train),
            np.array(plot_info.z_train_pred),
            title="Signed error distribution per depth slice (train)",
            savepath=(s_root / Path("error_violin_train.png")).as_posix(),
            show=show,
        )
        plots.plot_violin_depth_bins(
            np.array(plot_info.z_test),
            np.array(plot_info.z_test_pred),
            title="Signed error distribution per depth slice (val)",
            savepath=(s_root / Path("error_violin_val.png")).as_posix(),
            show=show,
        )

        plots.plot_hexbin_with_marginals(
            np.array(plot_info.z_train),
            np.array(plot_info.z_train_pred),
            title="Prediction density (train)",
            savepath=(s_root / Path("hexbin_train.png")).as_posix(),
            show=show,
        )
        plots.plot_hexbin_with_marginals(
            np.array(plot_info.z_test),
            np.array(plot_info.z_test_pred),
            title="Prediction density (val)",
            savepath=(s_root / Path("hexbin_val.png")).as_posix(),
            show=show,
        )


# NOTE: main database used ODP-DLHM-Database specs:
# 4640x3506 [px]
# 8bit-monochromatic
# 3.8 [um] pixel size
# 405, 510, 654, [nm] laser
# 0.5-4 [um]
@app.command()
def reconstruction(
    img_file_path: str = typer.Argument("", help="Path to image for reconstruction"),
    model_path: str = typer.Argument("best_model.pth", help="Path to trained model to use for torch optics analysis"),
    backbone: str = typer.Argument("efficientnet_b4", help="Model type being loaded"),
    crop_size: int = typer.Argument(512, help="Pixel width and height of image"),
    wavelength: float = typer.Argument(530e-9, help="Wavelength of light used to capture the image (m)"),
    z: float = typer.Argument(20e-3, help="Distance of measurement (m)"),
    dx: float = typer.Argument(1e-6, help="Size of image px (m)"),
):
    """Perform reconstruction on an hologram."""
    import holo.analysis.plots as plots  # performance reasons, import locally in function
    import holo.optics.reconstruction as rec
    from holo.analysis.metrics import error_metric

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

    typer.echo(f"the psnr is {psnr} with nrmsd: {nrmsd}")


@app.command()
def create_meta(hologram_directory: str, out_directory: str):
    """Build CSV containing metadata of images in hologram directory."""
    hologram_dir = Path(hologram_directory)
    out_dir = Path(out_directory)

    _ = build_metadata_csv(
        hologram_dir,
        out_dir,
    )


if __name__ == "__main__":
    app()
