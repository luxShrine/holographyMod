from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torchoptics as topo
from torchoptics.fields import Field

__all__ = ["fresnel_numpy", "propagate_torch"]


def propagate_torch(
    holo: torch.Tensor,
    *,
    wavelength: float,
    z: float,
    spacing: float = 3.8e-6,
    out: Literal["complex", "amp", "phase", "intensity"] = "complex",
) -> torch.Tensor:
    """Fresnel (angular-spectrum) propagation with TorchOptics.

    Args:
        holo:       torch.Tensor
                    Real-valued **square** tensor `[H,W]` or `[1,H,W]`, scaled 0‒1.
        wavelength: float
                    Wavelength in **metres** for this image.
        z:          float
                    Propagation distance in **metres** (positive = forward).
        spacing:    float, default 3.8 µm
                    Pixel pitch at the hologram plane (metres).
        out:        str, default "complex"
                    Which quantity to return:
                    `"complex"` (default), `"amp"`, `"phase"`, `"intensity"`.

    Returns:
        torch.Tensor: Complex field or real map, depending on out.

    """
    # TODO: <04-27-25>
    # normalise shape C×H×W
    if holo.dim() == 2:  # H×W → 1×H×W
        holo = holo.unsqueeze(0)
    elif holo.dim() != 3 or holo.size(0) != 1:
        raise ValueError("holo must be shape [H,W] or [1,H,W]")

    # TODO: <04-27-25>
    # convert amplitude-only tensor to complex field
    topo.set_default_spacing(spacing)  # global default (over-ridden below)
    field = Field(holo.sqrt(), z=0.0)  # amplitude = √intensity
    if not torch.is_complex(field.field):
        field.field = field.field.to(torch.complex64)

    # TODO: <04-27-25>
    # per-sample parameters
    field.wavelength = wavelength
    field.spacing[:] = spacing  # [dy, dx] broadcast

    # TODO: <04-27-25>
    # propagate
    reconstruct = field.propagate_to_z(z)  # returns new Field

    # TODO: <04-27-25>
    # select output
    match out:
        case "complex":
            return reconstruct.field.squeeze(0)  # -> H×W complex
        case "amp":
            return reconstruct.amplitude().squeeze(0)
        case "phase":
            return reconstruct.phase().squeeze(0)
        case "intensity":
            return reconstruct.intensity().squeeze(0)
        # TODO: <04-27-25>
        case _:
            raise ValueError(f"unknown out='{out}'")


# the transform to find the field
# E = h(x,y) * G(p,q) | p= x/(lambda z), q = y/(lambda z)
# TODO: what is a fast-ft? <04-16-25>
# TODO: explain the process of how to each step works mathematically
# Fourier Transform method of solving Fresnel Diffraction Integral
# G(p,q) = F(g(x,y)) = iint_{-inf}^{inf} g(x,y) exp(-i2 \pi (px+qy)) dx dy


# Fresnel transform:
def fresnel_numpy(img: npt.NDArray[np.float64], dx: float, wavelength: float, z: float) -> npt.NDArray[np.float64]:
    """Classical numperical reconstruction.

    Args:
        img: hologram as numpy array
        dx, dy: pixel's physical size [m]
        wavelength: wavelength of light used for imaging [m]
        z: propagation distance [m]
    returns:
        (amplitude, phase) at plane z.

    """
    # TODO: hologram squared hologram
    # hologram = Image.open(holo_filename)
    # hologram = crop_max_square(hologram)
    # convert to greyscale
    # hologram_array = np.asarray(hologram.convert("L"))

    # extract the amplitude and phase # TODO: ?

    # pixels in the x and y
    k = (2 * np.pi) / wavelength
    pixelX, pixelY = img.shape
    dy = dx

    # x, y coordinates, corresponding to pixels in image
    #
    # endpoint false as since we cross zero it acts as an additional index value, thus we remove that last endpoint
    # Further, these represent physical space, and thus must be scaled to that appropriate size per pixel
    x = np.linspace(-pixelX / 2, pixelX / 2, pixelX, endpoint=False) * dx
    y = np.linspace(-pixelY / 2, pixelY / 2, pixelY, endpoint=False) * dy
    X, Y = np.meshgrid(x, y)

    # apply the phase factor # TODO: ?
    # = exp((i \pi / \lambda z) (x^2 + y^2))
    phase_factor = np.exp((1j * np.pi * (X**2 + Y**2)) / (wavelength * z))
    image_pre = phase_factor * img

    # 2d ft
    # TODO: shift? np.fft.fftshift > ft2 > shift
    # image_transformed = np.fft.fft2(image_pre)
    image_transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_pre)))

    # in the process of converting from a continuous distribution (what the image is capturing)
    #
    # to a discrete set of samples (my arrays) we must limit the sampling rate, done here by: 0.5 * freq, where freq=(dn)^-1
    # Thus the transfer function h, needs its own grid
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), pixelX, endpoint=False)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), pixelY, endpoint=False)
    FX, FY = np.meshgrid(fx, fy)

    # transfer function  # TODO: ?
    transfer_func = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    # apply the transfer  # TODO: ?
    image_filtered = transfer_func * image_transformed

    # Inverse transform to solution
    #  TODO: shift? np.fft.fftshift > ift2 > shift
    image_reconstruct = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image_filtered)))

    # multiply by the phase factor again # TODO: ?
    # multiply by the scaling factor
    # scale_factor = np.exp(k / wavelength)
    scale_factor = np.exp(1j * k * z) / (1j * wavelength * z)
    image_final = image_reconstruct * scale_factor * phase_factor

    return image_final
