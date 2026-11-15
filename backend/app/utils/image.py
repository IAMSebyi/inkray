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


def decode_image_bytes(raw_bytes: bytes, max_image_size: int = 0) -> DecodedImage:
    if not raw_bytes:
        raise ValueError("Uploaded image file is empty.")

    with Image.open(BytesIO(raw_bytes)) as opened:
        if opened.format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format {opened.format!r}. Only JPEG and PNG are supported."
            )
        # Apply EXIF orientation so pixel coordinates match browser Image decoding.
        oriented = ImageOps.exif_transpose(opened)
        rgb_image = oriented.convert("RGB")
        orig_w, orig_h = rgb_image.size
        inf_w, inf_h = compute_inference_size(orig_w, orig_h, max_image_size)
        if (inf_w, inf_h) != (orig_w, orig_h):
            rgb_image = rgb_image.resize((inf_w, inf_h), Image.Resampling.LANCZOS)
        rgb = np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8))

    return DecodedImage(
        rgb=rgb,
        original_width=orig_w,
        original_height=orig_h,
        inference_width=inf_w,
        inference_height=inf_h,
    )
