from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision import transforms

from holo.train.autofocus import get_model
from holo.util.crop import crop_max_square

__all__ = ["fresnel_recon", "torch_recon"]


def torch_recon(
    img_file_path: str,
    wavelength: float,
    ckpt_file: str,
    crop_size: int = 512,
    z: float = 300e-6,
    backbone: str = "efficientnet_b4",
    spacing: float = 3e-6,
):
    """Fresnel.

    Args:
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
    pil_image: ImageType = Image.open(img_file_path).convert("RGB")
    pil_image_crop = crop_max_square(pil_image)
    np.asarray(crop_max_square(pil_image))

    # build architecture + load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)  # type: ignore
    bin_centers = ckpt["bin_centers"]

    model = get_model(ckpt["num_bins"], backbone).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # load & preprocess image
    preprocess = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            # use the same normalization you did in train
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = preprocess(pil_image_crop).unsqueeze(0).to(device)  # shape [1,C,H,W]# type: ignore

    # prediction
    with torch.no_grad():
        logits = model(x)  # shape [B, C]
        probs = torch.softmax(logits, dim=1)

        # discrete estimate
        # cls = probs.argmax(1)         # shape [B]
        # z_argmax = bin_centers[cls]   # depth in mm

        # continuous estimate
        z_expect = (probs * bin_centers).sum(1)  # ⟨z⟩ = Σ p_i z_i

    holo_gray = np.asarray(crop_max_square(pil_image).convert("L"), dtype=np.float32) / 255.0
    recon = fresnel_recon(holo_gray, dx=1.12e-6, wavelength=wavelength, z=float(z_expect) * 1e-3)
    amp: npt.NDArray[np.float64] = np.abs(recon)
    phase: npt.NDArray[Any] = np.angle(recon)

    hologram = np.array(pil_image_crop)
    return hologram, amp, phase


# the transform to find the field
# E = h(x,y) * G(p,q) | p= x/(lambda z), q = y/(lambda z)
# TODO: what is a fast-ft? <04-16-25>
# TODO: explain the process of how to each step works mathematically
# Fourier Transform method of solving Fresnel Diffraction Integral
# G(p,q) = F(g(x,y)) = iint_{-inf}^{inf} g(x,y) exp(-i2 \pi (px+qy)) dx dy


# Fresnel transform:
def fresnel_recon(img: npt.NDArray[np.float64], dx: float, wavelength: float, z: float) -> npt.NDArray[np.float64]:
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
