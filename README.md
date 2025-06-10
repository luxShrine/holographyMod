# holographyMod

Hologram autofocus trainer / evaluator

**Installation**

```console
git clone https://github.com/luxShrine/holographyMod.git
cd holographyMod
uv run holo --help # using uv
```

Or not using uv:

```console
python3 -m venv .venv # using native python
source .venv/bin/activate # bash
.venv\bin\activate # powershell

pip install -e . # install module to venv
```

**Setting Up Your Data:**

To use you own data you must provide a `.csv` file in the root of the dataset directory. The CSV file must contain the following:
- Column's with title's: Path;Wavelength;L_value;z_value
- Each column thus contains information corresponding to each image in the dataset:
    - Paths to each image relative to the CSV file 
    - Wavelength used in imaging (mm)
    - Distance from the point-source to the sample (mm)
    - True depth/z value, the distance from the sensor to the sample (mm)

If you do not have a CSV file with this information already setup, you can instead provide the information in the form of a text file. To do so, 
1. Organize the dataset by folders that contain shared properties
2. Create an `info.txt` file in a directory alongside the images, have it contain the expected fields. For example:
```
Wavelength = 500
L_value = 18.96
z_value = 0.521
```
3. The path will be gathered automatically, thus there is no need to provide it. Then run the train command with `--create-csv` to have the program create the CSV file and perform training.

**Usage**:

```console
holo [OPTIONS] COMMAND [ARGS]...
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

## `train`

Train the autofocus model based on supplied dataset.

**Usage**:

```console
holo train [OPTIONS] DS_ROOT
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

### Example Parameters

Quick test settings

```console
batch = 8
crop = 256
ep = 8
learn_rate = 1e-4
```

Substantial test settings

```console
batch = 16
crop = 512
ep = 50
learn_rate = 5e-5
```

## `plot-train`

Plot the data saved from autofocus training.

**Usage**:

```console
holo plot-train [OPTIONS]
```

**Options**:

- `-c, --classfiication`: Change analysis type to classification
- `--s`: Save the output plots, or display them. [default: True]
- `--help`: Show this message and exit.

## `reconstruction`

Perform reconstruction on an hologram.

**Usage**:

```console
holo reconstruction [OPTIONS] [IMG_FILE_PATH] [MODEL_PATH] [BACKBONE] [CROP_SIZE] [WAVELENGTH] [Z] [DX]
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
