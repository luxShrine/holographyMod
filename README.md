# `holographyMod`

Hologram autofocus trainer / evaluator

**Usage**:

```console
$ holographyMod [OPTIONS] COMMAND [ARGS]...
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
- `reconstruction`: Peform reconstruction on an hologram.
- `create-meta`: Build csv containing metadata of images in...

## `holographyMod train`

Train the autofocus model based on supplied dataset.

Args:

**Usage**:

```console
$ holographyMod train [OPTIONS] DS_ROOT
```

**Arguments**:

- `DS_ROOT`: [required]
  - Directory containing hologram images.

**Options**:

- `--metadata TEXT`: [default: ODP-DLHM-Database.csv]
  - Path to the metadata CSV file.
- `--out TEXT`: [default: checkpoints]
  - Directory to save checkpoints and logs.
- `--backbone TEXT`: [default: efficientnet_b4]
  - Model backbone name.
- `--crop INTEGER`: [default: 512]
  - Training batch size.
- `--value-split FLOAT`: [default: 0.2]
  - Size to crop images to.
- `--batch INTEGER`: [default: 16]
  - Number of training epochs.
- `--ep INTEGER`: [default: 10]
  - How fast should the model change epoch to epoch
- `--learn-rate FLOAT`: [default: 0.0001]
  - Fraction of data for validationw
- `--device-type TEXT`: [default: cuda]
  - Device ("cuda" or "cpu").
- `--help`: Show this message and exit.

## `holographyMod plot-train`

Plot the data saved from autofocus training.

**Usage**:

```console
$ holographyMod plot-train [OPTIONS]
```

**Options**:

- `--help`: Show this message and exit.

## `holographyMod reconstruction`

Peform reconstruction on an hologram.

Args:

**Usage**:

```console
$ holographyMod reconstruction [OPTIONS] IMG_FILE_PATH
```

**Arguments**:

- `IMG_FILE_PATH`: [required]
  - Path to image for reconstruction

**Options**:

- `--model-path TEXT`: [default: best_model.pth]
  - Path to trained model to use for torch optics anaylsis
- `--backbone TEXT`: [default: efficientnet_b4]
  - Model type being loaded
- `--crop-size INTEGER`: [default: 512]
  - Pixel width and height of image
- `--wavelength FLOAT`: [default: 5.3e-07]
  - Wavelength of light used to capture the image (m)
- `--z FLOAT`: [default: 0.02]
  - Distance of measurement (m)
- `--dx FLOAT`: [default: 1e-06]
  - Size of image px (m)
- `--help`: Show this message and exit.

## `holographyMod create-meta`

Build csv containing metadata of images in hologram directory.

**Usage**:

```console
$ holographyMod create-meta [OPTIONS] HOLOGRAM_DIRECTORY OUT_DIRECTORY
```

**Arguments**:

- `HOLOGRAM_DIRECTORY`: [required]
- `OUT_DIRECTORY`: [required]

**Options**:

- `--help`: Show this message and exit.
