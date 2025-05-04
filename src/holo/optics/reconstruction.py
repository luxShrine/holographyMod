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
    crop_size: int = 256,
    z: float = 300e-6,  #  TODO: implement both predictions, maybe allow choosing
    backbone: str = "efficientnet_b4",
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

    Returns:
        Hologram, Amplitude, phase, all as numpy arrays

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
        # cls = probs.argmax(1) # shape [B]
        # z_argmax = bin_centers[cls] # depth in mm

        # continuous estimate
        z_expect = (probs * bin_centers).sum(1)  # ⟨z⟩ = Σ p_i z_i

    # TODO: why float32?
    holo_gray: npt.NDArray[np.float32] = np.asarray(crop_max_square(pil_image).convert("L"), dtype=np.float32) / 255.0
    recon = fresnel_recon(holo_gray, dx=1e-6, wavelength=wavelength, z=float(z_expect) * 1e-3)
    amp: npt.NDArray[np.float64] = np.abs(recon)
    phase: npt.NDArray[Any] = np.angle(recon)

    hologram = np.array(pil_image_crop)
    return hologram, amp, phase


# the transform to find the field
# E = h(x,y) * G(p,q) | p= x/(lambda z), q = y/(lambda z)

# Fourier Transform method of solving Fresnel Diffraction Integral
# G(p,q) = F(g(x,y)) = iint_{-inf}^{inf} g(x,y) exp(-i2 \pi (px+qy)) dx dy

# TODO: what is a fast-ft? <04-16-25>
# TODO: explain the process of how to each step works mathematically


# Fresnel transform:
def fresnel_recon(img: npt.NDArray[np.float32], dx: float, wavelength: float, z: float) -> npt.NDArray[np.float64]:
    """Classical numperical reconstruction via Fourier transform.

    Args:
        img: hologram as numpy array
        dx: pixel's physical size [m]
        wavelength: wavelength of light used for imaging [nm]
        z: propagation distance [um]
    returns:
        (amplitude, phase) at plane z.

    """
    k = (2 * np.pi) / wavelength
    pixelX, pixelY = img.shape  # returns the dimensions of image array, x, y
    dy = dx  # pixels assumed be of equal length/width

    # create array, with each value of x, y, corresponding to pixels in image
    # endpoint=false since we cross zero it acts as an additional index value, remove that last endpoint
    # to make this physical, must scale by size per pixel
    x = np.linspace(-pixelX / 2, pixelX / 2, pixelX, endpoint=False) * dx
    y = np.linspace(-pixelY / 2, pixelY / 2, pixelY, endpoint=False) * dy
    X, Y = np.meshgrid(x, y)  # fills in the space between these two "basis" vectors

    # apply the phase factor # TODO: ?
    # = exp((i \pi / \lambda z) (x^2 + y^2))
    phase_factor = np.exp((1j * np.pi * (X**2 + Y**2)) / (wavelength * z))
    image_pre = phase_factor * img

    # 2d ft
    # TODO: shift? np.fft.fftshift > ft2 > shift
    # image_transformed = np.fft.fft2(image_pre)
    image_transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_pre)))

    # in the process of converting from a continuous distribution (what the image is capturing)
    # to the discrete arrays we must limit the sampling rate, by: 0.5 * freq, where freq=(dn)^-1
    # using the transfer function, which needs its own grid
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
