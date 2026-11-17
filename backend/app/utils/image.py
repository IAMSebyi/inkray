from dataclasses import dataclass
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps

from ..geometry.transforms import compute_inference_size

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(ValueError):
    pass


def validate_image_pixel_count(width: int, height: int, max_pixels: int) -> None:
    if width * height > max_pixels:
        raise ImageTooLargeError("Uploaded image exceeds the pixel count limit.")


def decode_image_bytes(
    raw_bytes: bytes,
    max_image_size: int = 0,
    max_pixels: int = 64_000_000,
) -> DecodedImage:
    if not raw_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    try:
        with Image.open(BytesIO(raw_bytes)) as opened:
            if opened.format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError(
                    f"Unsupported image format {opened.format!r}. Only JPEG and PNG are supported."
                )
            validate_image_pixel_count(opened.width, opened.height, max_pixels)
            # Apply EXIF orientation so pixel coordinates match browser Image decoding.
            oriented = ImageOps.exif_transpose(opened)
            rgb_image = oriented.convert("RGB")
            orig_w, orig_h = rgb_image.size
            inf_w, inf_h = compute_inference_size(orig_w, orig_h, max_image_size)
            if (inf_w, inf_h) != (orig_w, orig_h):
                rgb_image = rgb_image.resize((inf_w, inf_h), Image.Resampling.LANCZOS)
            rgb = np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8))
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("Uploaded image exceeds the pixel count limit.") from exc
    except OSError as exc:
        raise InvalidImageError("Uploaded image could not be decoded.") from exc

    return DecodedImage(
        rgb=rgb,
        original_width=orig_w,
        original_height=orig_h,
        inference_width=inf_w,
        inference_height=inf_h,
    )
