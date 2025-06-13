from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import holo.infra.training as ts
from holo.infra.util.image_processing import crop_max_square
from holo.infra.util.types import AnalysisType, Np1Array32

if TYPE_CHECKING:
    from PIL.Image import Image as ImageType

__all__ = ["recon_inline", "torch_recon"]


def torch_recon(
    img_file_path: str,
    wavelength: float,
    ckpt_file: str,
    crop_size: int = 512,
    z: float = 300e-6,
    backbone: str = "efficientnet_b4",
    dx: float = 3.8e-6,
):
    """Fresnel.

    Args:
        img_file_path: str,
                       Path to image to reconstruct.
        wavelength:    float,
                       Wavelength for this image (m).
        ckpt_file:     str,
                       path to the of weights to use in the model.
        crop_size:     int,
                       Size the length/width to crop the image to.
        z:             float,
                       propagation distance (m) .
        backbone:      str,
                       Name of the pre-trained model to apply the weights to.

        dx:            float,
                       Size of pixel width (m).

    Returns:
        Hologram, Amplitude, phase, all as numpy arrays

    """
    pil_image: ImageType = Image.open(img_file_path).convert("RGB")
    pil_image_crop = crop_max_square(pil_image)
    _np_image = np.asarray(crop_max_square(pil_image))

    # build architecture + load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## WARN: This is iffy
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    bin_centers = cast("list[float]", ckpt["bin_centers"])

    num_classes: int = cast("int", ckpt["num_bins"])
    auto_method: AnalysisType = AnalysisType.CLASS if num_classes != 1 else AnalysisType.REG
    model: nn.Module = ts.get_model(backbone, num_classes, auto_method).to(device)
    _ = model.load_state_dict(ckpt["model_state_dict"])
    _ = model.eval()

    # load & preprocess image
    preprocess_trans = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.Grayscale(num_output_channels=3),  # WARN: needed?
            transforms.ToTensor(),
            # use the same normalization in train
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # shape [1, C, H, W]
    x = preprocess_trans(pil_image_crop).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        logits = model(x)  # shape [B, C]
        probs = torch.softmax(logits, dim=1)  # rescales elements for range [0,1] & sum to one

        # discrete estimate
        cls = probs.argmax(1)  # shape [B]
        z_argmax = bin_centers[cls]  # depth in mm
        z_expect = z_argmax  # ⟨z⟩ = Σ p_i z_i

        # continuous estimate
        # z_expect = (probs * bin_centers).sum(1)  # ⟨z⟩ = Σ p_i z_i

    # torch expects float32
    holo_gray = np.asarray(crop_max_square(pil_image).convert("L"), np.float32) / 255.0
    amp, phase = recon_inline(holo_gray, wavelength=wavelength, z=float(z_expect) * 1e-3, px=dx)

    hologram = np.array(pil_image_crop)
    return hologram, amp, phase


def recon_inline(intensity: Np1Array32, wavelength: float, z: float, px: float):
    r"""Use Fourier transform method to return reconstructed amplitdude and phase of image.

    Args:
        intensity: type and description.
        lamb: wavelength of light used.
        z: propagation distance.
        px: pixel size.

    Returns:
        Phase and amplitude of reconstructed image.

    Detail:
        The transform to find the field
        E = h(x,y) * G(p,q) | p= x/(lambda z), q = y/(lambda z)
        Fourier Transform method of solving Fresnel Diffraction Integral
        G(p,q) = F(g(x,y)) = iint_{-inf}^{inf} g(x,y) exp(-i2 \pi (px+qy)) dx dy

    """
    # image captures just the intensity of the object, take as complex sqrt since it is a 2D wave
    field0 = np.sqrt(intensity).astype(np.complex64)

    k = (2 * np.pi) / wavelength  # wavenumber
    pixelX, pixelY = field0.shape  # returns the dimensions of image array, store them
    # create arrays of size defined by image dimensions, containing sample frequencies
    fx = np.fft.fftfreq(pixelX, d=px)
    fy = np.fft.fftfreq(pixelY, d=px)
    FX, FY = np.meshgrid(fx, fy)  # fills in the space between these two "basis" vectors

    # in the process of converting from a continuous distribution (what the image is capturing)
    # to the discrete arrays we must limit the sampling rate
    ikz = 1j * k * z
    H = np.exp(np.sqrt(ikz - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
    # find the 2D fft of the field, -> apply phase -> inverse 2D fft to result
    # using the phase factor H
    U1 = np.fft.ifft2(np.fft.fft2(field0) * H)
    amplitdude = np.abs(U1)
    phase = np.angle(U1)
    return amplitdude, phase
