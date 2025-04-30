import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.image import AxesImage
from PIL import Image
from PIL.Image import Image as ImageType

from holo.io.paths import data_root
from holo.optics.reconstruction import fresnel_numpy
from holo.util.crop import crop_max_square


def main_recon():
    # Values needed
    wavelength = 530e-9
    z = 300e-6  # um
    dx = 1.12e-6  # size of image px

    img_file = data_root() / "DHM_1" / "DHM" / "Static" / "Brain tissue" / "holo.png"
    pil_image: ImageType = Image.open(img_file)

    cropped_image = np.asarray(crop_max_square(pil_image))

    image_raw = fresnel_numpy(cropped_image, dx, wavelength, z)

    amplitude = np.abs(image_raw)
    phase = np.angle(image_raw)

    # temp plotting to see it works
    fig, axs = plt.subplots(2, 2)

    plot_amp: AxesImage = axs[0, 0].imshow(amplitude, cmap="gray")
    axs[0, 0].set_title("Reconstructed Amplitude")
    fig.colorbar(plot_amp, ax=axs[0, 0])
    # plt.show()

    plot_phase = axs[0, 1].imshow(phase, cmap="gray")
    axs[0, 1].set_title("Reconstructed Phase")
    fig.colorbar(plot_phase, ax=axs[0, 1])
    # plt.show()

    hologram = Image.open(img_file)
    plot_holo = axs[1, 0].imshow(hologram, cmap="gray")
    axs[1, 0].set_title("Original Image Phase")
    fig.colorbar(plot_holo, ax=axs[1, 0])

    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(main_recon)
