# CLI

Hologram autofocus trainer / evaluator

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

- `-v, --verbose`: Print DEBUG messages to the terminal
- `--log-file PATH`: Path to the log file [default: debug.log]
- `--install-completion`: Install completion for the current shell.
- `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
- `--help`: Show this message and exit.

**Commands**:

- `train`: Train the autofocus model based on...
- `plot-train`: Plot the data saved from autofocus training.
- `reconstruction`: Perform reconstruction on an hologram.
- `create-meta`: Build CSV containing metadata of images in...

## `train`

Train the autofocus model based on supplied dataset.

**Usage**:

```console
$ train [OPTIONS] DS_ROOT
```

**Arguments**:

- `DS_ROOT`: Directory containing hologram images. [required]

**Options**:

- `--meta TEXT`: Path to the metadata CSV file. [default: ODP-DLHM-Database.csv]
- `--out TEXT`: Directory to save checkpoints and logs. [default: checkpoints]
- `--backbone, --bb TEXT`: Model backbone name. [default: efficientnet_b4]
- `--crop, --c INTEGER`: Size to crop images to. [default: 512]
- `--vs FLOAT`: Fraction of data for validation. [default: 0.2]
- `--batch, --ba INTEGER`: Training batch size. [default: 16]
- `--ep INTEGER`: Number of training epochs. [default: 10]
- `--lr FLOAT`: How fast should the model change epoch to epoch [default: 0.0001]
- `--device TEXT`: Device ("cuda" or "cpu"). [default: cuda]
- `-c, --classfiication`: Change analysis type to classification
- `--help`: Show this message and exit.

## `plot-train`

Plot the data saved from autofocus training.

**Usage**:

```console
$ plot-train [OPTIONS]
```

**Options**:

- `-c, --classfiication`: Change analysis type to classification
- `--help`: Show this message and exit.

## `reconstruction`

Perform reconstruction on an hologram.

**Usage**:

```console
$ reconstruction [OPTIONS] [IMG_FILE_PATH] [MODEL_PATH] [BACKBONE] [CROP_SIZE] [WAVELENGTH] [Z] [DX]
```

**Arguments**:

- `[IMG_FILE_PATH]`: Path to image for reconstruction [default: best_model.pth]
- `[MODEL_PATH]`: Path to trained model to use for torch optics analysis [default: best_model.pth]
- `[BACKBONE]`: Model type being loaded [default: efficientnet_b4]
- `[CROP_SIZE]`: Pixel width and height of image [default: 512]
- `[WAVELENGTH]`: Wavelength of light used to capture the image (m) [default: 5.3e-07]
- `[Z]`: Distance of measurement (m) [default: 0.02]
- `[DX]`: Size of image px (m) [default: 1e-06]

**Options**:

- `--help`: Show this message and exit.

## `create-meta`

Build CSV containing metadata of images in hologram directory.

**Usage**:

```console
$ create-meta [OPTIONS] HOLOGRAM_DIRECTORY OUT_DIRECTORY
```

**Arguments**:

- `HOLOGRAM_DIRECTORY`: [required]
- `OUT_DIRECTORY`: [required]

**Options**:

- `--help`: Show this message and exit.
