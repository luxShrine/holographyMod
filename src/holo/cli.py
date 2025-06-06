import logging
from pathlib import Path
from typing import Any

import click
import numpy.typing as npt

# import typer
from holo.infra.dataclasses import PlotPred, load_obj, save_obj
from holo.infra.log import init_logging
from holo.infra.util.image_processing import build_metadata_csv
from holo.infra.util.paths import MW_data, path_check, static_root
from holo.infra.util.types import AnalysisType, DisplayType, UserDevice

# Must be called before anything logs
init_logging()


DEF_IMG_FILE_PATH = (MW_data() / "510" / "10_Phase_USAF" / "z10" / "10.jpg").as_posix()


@click.group()
def cli():
    pass


# TODO: load settings from external file rather than from cli
@cli.command()
@click.argument("ds_root")
@click.option(
    "--meta_csv_name", default="ODP-DLHM-Database.csv", help="Path to the metadata CSV file."
)
@click.option("--num_classes", default=10, help="Number of classifications.")
@click.option("--out_dir", default="checkpoints", help="Directory to save checkpoints and logs.")
@click.option("--backbone", default="efficientnet_b4", help="Model backbone name.")
@click.option("--crop_size", default=224, help="Size to crop images to.")
@click.option("--val_split", default=0.2, help="Fraction of data for validation.")
@click.option("--batch_size", default=16, help="Training batch size.")
@click.option("--ep", default=10, help="Number of training epochs.")
@click.option("--opt_lr", default=5e-5, help="How fast should the model change epoch to epoch")
@click.option("--device_user", default="cuda", help="Device ('cuda' or 'cpu').")
@click.option("--analysis", default=True, help="Change analysis type to classification")
@click.option("--seed", default=True, help="Keep the random seed consistent.")
@click.option("--cont_train", default=False, help="Continue from checkpoint")
def train(
    ds_root: Path,
    meta_csv_name: str,
    num_classes: int,
    out_dir: str,
    backbone: str,
    crop_size: int,
    val_split: float,
    batch_size: int,
    ep: int,
    opt_lr: float,
    device_user: str,
    analysis: bool,
    seed: bool,
    cont_train: bool,
) -> None:
    """Train the autofocus model based on supplied dataset."""
    from holo.infra.dataclasses import AutoConfig
    from holo.infra.training import train_autofocus

    logger = logging.getLogger(__name__)

    auto_method = AnalysisType.CLASS if analysis else AnalysisType.REG

    logger.info(f"Training type: {auto_method}.")

    path_ckpt: str | None = None if cont_train is False else "./checkpoints/latest_checkpoint.tar"

    dev = UserDevice.CUDA if device_user == "cuda" else UserDevice.CPU

    # WARN: not robust for all types, will need a change <05-09-25>
    if backbone == "vit" and crop_size != 224:
        logger.error(
            f"backbone of type {backbone} requires a crop size of 224,"
            + "defaulting to appropriate crop size"
        )
        crop_size = 224

    if (ds_root / Path(meta_csv_name)).exists():
        meta_csv_strpath = (ds_root / Path(meta_csv_name)).as_posix()
        logger.info(f" path to csv exists {meta_csv_name}")
    else:
        new_def_path: Path = MW_data() / Path(meta_csv_name)
        logger.error(f" path to csv does not exist, using default paths {new_def_path}")
        logger.warning(f"Checking {new_def_path}")
        if new_def_path.exists():
            meta_csv_strpath = (new_def_path).as_posix()
            logger.info(f"{meta_csv_strpath} exists, continuing...")
        else:
            raise Exception(f" path to csv still does not exist: {new_def_path}")

    autofocus_config = AutoConfig(
        out_dir=out_dir,
        meta_csv_strpath=meta_csv_strpath,
        num_classes=num_classes,
        backbone=backbone,
        batch_size=batch_size,
        epoch_count=ep,
        crop_size=crop_size,
        opt_lr=opt_lr,
        val_split=val_split,
        device_user=dev,
        analysis=auto_method,
        fixed_seed=seed,
    )

    plot_info = train_autofocus(autofocus_config, path_ckpt)
    save_obj(plot_info)


@cli.command()
@click.option("--analysis", default=False, help="Change analysis type to classification")
@click.option("--display", default="save", help="Save the output plots, show them, or both.")
def plot_train(
    analysis: bool,
    display: DisplayType,
):
    """Plot the data saved from autofocus training."""
    import holo.core.plots as plots  # performance reasons, import locally in function

    logger = logging.getLogger(__name__)

    logger.info("plotting training data...")

    plot_info = load_obj()[0]
    assert isinstance(plot_info, PlotPred), f"plot_info is not PlotPred, found {type(plot_info)}"
    # update plot obj with desired values
    logger.info(f"Plotting function with option: {display}")
    s_root = static_root().as_posix()
    plot_info.display = display

    # TODO: automatically get analysis type
    if analysis:
        plots.plot_actual_versus_predicted(
            plot_info,
            title="Classification Residual",
            path_to_plot=(s_root / Path("class_residual_vs_true_train.png")).as_posix(),
        )
    else:
        plots.plot_residual_vs_true(
            plot_info,
            title="Residual vs True depth (train)",
            path_to_plot=(s_root / Path("residual_vs_true_train.png")).as_posix(),
        )
        plots.plot_residual_vs_true(
            plot_info,
            title="Residual vs True depth (val)",
            path_to_plot=(s_root / Path("residual_vs_true_val.png")).as_posix(),
        )

        plots.plot_violin_depth_bins(
            plot_info,
            title="Signed error distribution per depth slice (train)",
            path_to_plot=(s_root / Path("error_violin_train.png")).as_posix(),
        )
        plots.plot_violin_depth_bins(
            plot_info,
            title="Signed error distribution per depth slice (val)",
            path_to_plot=(s_root / Path("error_violin_val.png")).as_posix(),
        )

        plots.plot_hexbin_with_marginals(
            plot_info,
            title="Prediction density (train)",
            path_to_plot=(s_root / Path("hexbin_train.png")).as_posix(),
        )
        plots.plot_hexbin_with_marginals(
            plot_info,
            title="Prediction density (val)",
            path_to_plot=(s_root / Path("hexbin_val.png")).as_posix(),
        )


# NOTE: main database used ODP-DLHM-Database specs:
# 4640x3506 [px]
# 8bit-monochromatic
# 3.8 [um] pixel size
# 405, 510, 654, [nm] laser
# 0.5-4 [um]


@cli.command()
@click.argument("img_file_path")
@click.option("--amp_true", default=None, help="True amplitude")
@click.option("--phase_true", default=None, help="True Phase")
@click.option(
    "--model_path",
    default="best_model.pth",
    help="Path to trained model to use for torch optics analysis",
)
@click.option("--backbone", default="efficientnet_b4", help="Model type being loaded")
@click.option("--crop_size", default=512, help="Pixel width and height of image")
@click.option(
    "--wavelength", default=530e-9, help="Wavelength of light used to capture the image (m)"
)
@click.option("--z", default=20e-3, help="Distance of measurement (m)")
@click.option("--dx", default=1e-6, help="Size of image px (m)")
@click.option(
    "--display",
    default=DisplayType.SAVE,
    help="Show, Save or do both for resulting phase and amplitude images",
)
def reconstruction(
    img_file_path: str,
    model_path: str,
    backbone: str,
    crop_size: int,
    wavelength: float,
    z: float,
    dx: float,
    display: DisplayType,
    amp_true: None | npt.NDArray[Any],
    phase_true: None | npt.NDArray[Any],
):
    """Perform reconstruction on an hologram."""
    import holo.core.plots as plots  # performance reasons, import locally in function

    logger = logging.getLogger(__name__)

    logger.info("Starting reconstruction...")

    ckpt_file = Path("checkpoints") / model_path
    path_check(checkpoint_file=ckpt_file, img_file_path=Path(img_file_path))

    # perform reconstruction
    plots.plot_amp_phase(
        backbone=backbone,
        crop_size=crop_size,
        ckpt_file=ckpt_file,
        display=display,
        dx=dx,
        img_file_path=img_file_path,
        wavelength=wavelength,
        z=z,
        amp_true=amp_true,
        phase_true=phase_true,
    )


@cli.command()
def create_meta(hologram_directory: str, out_directory: str):
    """Build CSV containing metadata of images in hologram directory."""
    logger = logging.getLogger(__name__)

    logger.info("Creating proper CSV...")
    hologram_dir = Path(hologram_directory)
    out_dir = Path(out_directory)

    _ = build_metadata_csv(
        hologram_dir,
        out_dir,
    )


cli.add_command(train)
if __name__ == "__main__":
    cli()
