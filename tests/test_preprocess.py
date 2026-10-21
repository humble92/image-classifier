import io

import pytest
from PIL import Image

from image_classifier.config import (
    JEV_MAX_PIXELS,
    JEV_MAX_SIDE,
    JEV_MIN_PIXELS,
    JEV_MIN_SIDE,
)
from image_classifier.preprocess import JevImageError, fit_jev_size, jev_jpeg, open_rgb, validate_jev_image


def _dims(data: bytes):
    with Image.open(io.BytesIO(data)) as im:
        return im.format, im.size


@pytest.mark.parametrize(
    "size",
    [(4000, 3000), (3000, 4000), (100, 80), (280, 280), (1024, 1024), (1600, 200), (5000, 100), (50, 3000), (1, 1)],
)
@pytest.mark.parametrize("max_side", [280, 1024, 1568])
def test_jev_jpeg_within_bounds(tmp_path, size, max_side):
    p = tmp_path / "im.png"
    Image.new("RGB", size, (200, 10, 10)).save(p)
    fmt, (w, h) = _dims(jev_jpeg(p, max_side=max_side))
    assert fmt == "JPEG"
    assert JEV_MIN_SIDE <= w <= JEV_MAX_SIDE and JEV_MIN_SIDE <= h <= JEV_MAX_SIDE
    assert max(w, h) <= max_side
    assert JEV_MIN_PIXELS <= w * h <= JEV_MAX_PIXELS


def test_fit_keeps_aspect_when_possible():
    (rw, rh), canvas = fit_jev_size(4000, 3000, 1024)
    assert (rw, rh) == (1024, 768) and canvas == (1024, 768)
    (rw, rh), canvas = fit_jev_size(100, 50, 1024)
    assert (rw, rh) == (560, 280) and canvas == (560, 280)


def test_fit_pads_extreme_aspect():
    (rw, rh), (cw, ch) = fit_jev_size(5000, 100, 1024)
    assert rw == 1024 and rh < JEV_MIN_SIDE
    assert (cw, ch) == (1024, JEV_MIN_SIDE)


def test_exif_orientation_applied(tmp_path):
    p = tmp_path / "rot.jpg"
    im = Image.new("RGB", (400, 300))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    im.save(p, exif=exif)
    assert open_rgb(p).size == (300, 400)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "L", "CMYK", "I;16"])
def test_open_rgb_modes(tmp_path, mode):
    p = tmp_path / ("im.tif" if mode in ("CMYK", "I;16") else "im.png")
    Image.new(mode, (300, 300)).save(p)
    im = open_rgb(p)
    assert im.mode == "RGB"
    validate_jev_image(jev_jpeg(p))


def test_validate_rejects_bad_images():
    def png(size):
        b = io.BytesIO()
        Image.new("RGB", size).save(b, "PNG")
        return b.getvalue()

    validate_jev_image(png((280, 280)))
    for bad in [(279, 400), (400, 279), (1569, 400)]:
        with pytest.raises(JevImageError):
            validate_jev_image(png(bad))
    b = io.BytesIO()
    Image.new("RGB", (300, 300)).save(b, "WEBP")
    with pytest.raises(JevImageError, match="format"):
        validate_jev_image(b.getvalue())
    with pytest.raises(JevImageError, match="decode"):
        validate_jev_image(b"garbage")
