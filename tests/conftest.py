from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_classifier.config import Config
from image_classifier.embed import l2_normalize


class FakeEmbedder:
    """Deterministic stand-in for DINOv2: mean-centered 8x8 RGB thumbnail."""

    tag = "fake-v1"

    def __init__(self):
        self.calls = 0
        self.images_seen = 0

    def embed(self, images):
        self.calls += 1
        self.images_seen += len(images)
        rows = []
        for im in images:
            a = np.asarray(im.convert("RGB").resize((8, 8), Image.Resampling.BOX), dtype=np.float32).ravel()
            rows.append(a - a.mean())
        return l2_normalize(np.stack(rows))


def pattern(seed: int, size=(320, 320), grid: int = 8) -> np.ndarray:
    """An 'identity': a random grid of colored blocks."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(grid, grid, 3)).astype(np.float32)


def render(blocks: np.ndarray, size=(320, 320), noise: float = 0.0, seed: int = 0, gain: float = 1.0) -> Image.Image:
    im = Image.fromarray(blocks.clip(0, 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.float32) * gain
    if noise:
        a += np.random.default_rng(seed).normal(0, noise, a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return a * w + b * (1 - w)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.embed.cache_dir = str(tmp_path / "cache")
    c.embed.flip_refs = False  # block patterns are not flip-invariant
    # Fixture scores: duplicates ~1.0, look-alikes ~0.81-0.85, A/B mix ~0.70/0.72, unrelated < 0.11.
    c.band.t_high, c.band.t_low, c.band.margin = 0.90, 0.40, 0.05
    return c


@pytest.fixture
def dataset(tmp_path: Path):
    """2 references, ~20 candidates: near-duplicates, look-alikes, unrelated, broken."""
    refs_dir = tmp_path / "refs"
    inp = tmp_path / "input"
    (inp / "sub").mkdir(parents=True)
    refs_dir.mkdir()
    A, B = pattern(1), pattern(2)
    render(A).save(refs_dir / "red bike.png")
    render(B).save(refs_dir / "blue_bike.jpg", quality=95)

    expected: dict[str, str] = {}

    def add(name: str, im: Image.Image, folder: str, sub: bool = False):
        path = (inp / "sub" if sub else inp) / name
        im.save(path, quality=92) if path.suffix == ".jpg" else im.save(path)
        expected[str(path)] = folder

    for i in range(4):  # near-duplicates of A and B (noise, brightness, scale)
        add(f"a_dup{i}.jpg", render(A, size=(400 + 40 * i, 300), noise=6, seed=i, gain=0.95 + 0.03 * i), "red_bike")
        add(f"b_dup{i}.png", render(B, size=(320, 320), noise=6, seed=10 + i), "blue_bike", sub=i % 2 == 0)
    for i in range(6):  # unrelated
        add(f"other{i}.jpg", render(pattern(100 + i), noise=4, seed=20 + i), "_unmatched")
    for i in range(4):  # look-alikes: blends that sit between bands -> borderline
        add(f"look_a{i}.png", render(blend(A, pattern(200 + i), 0.6)), "BORDERLINE")
    add("ab_mix.png", render(blend(A, B, 0.5)), "BORDERLINE")  # similar to both -> borderline
    # A byte-identical copy of reference A goes straight to its folder.
    (inp / "copy_of_ref.png").write_bytes((refs_dir / "red bike.png").read_bytes())
    expected[str(inp / "copy_of_ref.png")] = "red_bike"
    (inp / "broken.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    (inp / "notes.txt").write_text("ignored")
    return {"refs": refs_dir, "input": inp, "expected": expected, "out": tmp_path / "out"}
