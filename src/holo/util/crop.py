from PIL.Image import Image as ImageType


def crop_center(pil_img: ImageType, crop_width: int, crop_height: int) -> ImageType:
    """Crop provided image into around its center."""
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


def crop_max_square(pil_img: ImageType) -> ImageType:
    """Find the dimensions of image, crop to the largest square around its center."""
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))
