"""Image decoding and the Jev-compliant JPEG derivative."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from .config import (
    JEV_MAX_IMAGE_BYTES,
    JEV_MAX_PIXELS,
    JEV_MAX_SIDE,
    JEV_MIN_PIXELS,
    JEV_MIN_SIDE,
)

PAD_COLOR = (128, 128, 128)


class JevImageError(ValueError):
    """An image that would be rejected by Jev (and fail the whole request)."""


def open_rgb(path: Path) -> Image.Image:
    """Decode, apply EXIF orientation, flatten alpha onto white, return RGB."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            return bg
        return im.convert("RGB")


def fit_jev_size(w: int, h: int, max_side: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((resized_w, resized_h), (canvas_w, canvas_h)).

    The image is scaled so its long side is <= max_side and its short side is >= 280.
    When the aspect ratio makes both impossible (ratio > max_side/280), the long side
    is capped and the short side is padded up to 280 on a canvas.
    """
    long_side, short_side = max(w, h), min(w, h)
    scale = min(1.0, max_side / long_side)
    if short_side * scale < JEV_MIN_SIDE:
        scale = JEV_MIN_SIDE / short_side
        if long_side * scale > max_side:
            scale = max_side / long_side
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    # Rounding can push a side 1 px past a bound; clamp.
    rw, rh = min(rw, max_side), min(rh, max_side)
    cw, ch = max(rw, JEV_MIN_SIDE), max(rh, JEV_MIN_SIDE)
    return (rw, rh), (cw, ch)


def jev_jpeg(path: Path, max_side: int = 1024, quality: int = 90) -> bytes:
    im = open_rgb(path)
    (rw, rh), (cw, ch) = fit_jev_size(im.width, im.height, max_side)
    if (rw, rh) != im.size:
        im = im.resize((rw, rh), Image.Resampling.LANCZOS)
    if (cw, ch) != (rw, rh):
        canvas = Image.new("RGB", (cw, ch), PAD_COLOR)
        canvas.paste(im, ((cw - rw) // 2, (ch - rh) // 2))
        im = canvas
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    validate_jev_image(data)
    return data


def validate_jev_image(data: bytes) -> None:
    """Raise JevImageError if ``data`` violates any documented Jev image limit."""
    if len(data) > JEV_MAX_IMAGE_BYTES:
        raise JevImageError(f"image is {len(data)} bytes, limit {JEV_MAX_IMAGE_BYTES}")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            n_frames = getattr(im, "n_frames", 1)
    except Exception as exc:
        raise JevImageError(f"cannot decode: {exc}") from exc
    if fmt not in ("JPEG", "PNG"):
        raise JevImageError(f"format {fmt} not allowed (PNG or JPEG only)")
    if n_frames != 1:
        raise JevImageError("animated images are not allowed")
    if not (JEV_MIN_SIDE <= w <= JEV_MAX_SIDE and JEV_MIN_SIDE <= h <= JEV_MAX_SIDE):
        raise JevImageError(f"size {w}x{h} outside {JEV_MIN_SIDE}-{JEV_MAX_SIDE} px")
    if not (JEV_MIN_PIXELS <= w * h <= JEV_MAX_PIXELS):
        raise JevImageError(f"{w * h} pixels outside {JEV_MIN_PIXELS}-{JEV_MAX_PIXELS}")
