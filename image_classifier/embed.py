"""DINOv2 / SigLIP embedding with an on-disk cache.

torch and transformers are imported lazily so the rest of the package (and most
tests) work without them.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .config import EmbedConfig
from .io_scan import ImageFile
from .preprocess import open_rgb

log = logging.getLogger(__name__)

MODEL_IDS = {
    "dinov2-base": "facebook/dinov2-base",
    "siglip-base": "google/siglip-base-patch16-224",
}


class Embedder(Protocol):
    tag: str  # identifies model + feature variant; used as the cache namespace

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return an (len(images), D) float32 array of L2-normalized rows."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class HFEmbedder:
    """Wraps a Hugging Face vision backbone (DINOv2 or SigLIP vision tower)."""

    def __init__(self, model, processor, kind: str, tag: str, patch_mean: bool = False, device: str = "cpu"):
        if kind not in ("dinov2", "siglip"):
            raise ValueError(f"unknown model kind {kind}")
        self.model = model.to(device).eval()
        self.processor = processor
        self.kind = kind
        self.patch_mean = patch_mean
        self.device = device
        self.tag = tag + ("+patchmean" if patch_mean else "")

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=pixel_values)
            hidden = out.last_hidden_state
            if self.kind == "dinov2":
                # last_hidden_state is already layer-normed; token 0 is CLS.
                feats = hidden[:, 0]
                patches = hidden[:, 1:]
            else:
                # SigLIP has no CLS token; its pooled output is the attention-pool head.
                feats = out.pooler_output
                patches = hidden
            if self.patch_mean:
                feats = torch.cat([feats, patches.mean(dim=1)], dim=-1)
        return l2_normalize(feats.float().cpu().numpy())


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedder(cfg: EmbedConfig) -> HFEmbedder:
    """Load pretrained weights (downloads ~350 MB on first run, then cached by HF)."""
    if cfg.model not in MODEL_IDS:
        raise ValueError(f"unknown model {cfg.model}; choose from {sorted(MODEL_IDS)}")
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

    model_id = MODEL_IDS[cfg.model]
    processor = AutoImageProcessor.from_pretrained(model_id)
    if cfg.model.startswith("siglip"):
        model, kind = SiglipVisionModel.from_pretrained(model_id), "siglip"
    else:
        model, kind = AutoModel.from_pretrained(model_id), "dinov2"
    return HFEmbedder(model, processor, kind, cfg.model, cfg.patch_mean, resolve_device(cfg.device))


class EmbeddingCache:
    """``<cache_dir>/<tag>/<sha256><suffix>.npy``; one vector per file."""

    def __init__(self, cache_dir: Path | str, tag: str):
        safe_tag = "".join(c if c.isalnum() or c in "+-_." else "_" for c in tag)
        self.dir = Path(cache_dir) / safe_tag
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return np.load(p, allow_pickle=False)
        except (OSError, ValueError):
            log.warning("corrupt cache entry %s; recomputing", p)
            return None

    def put(self, key: str, vec: np.ndarray) -> None:
        # Write to a temp file then rename so a crash never leaves a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, vec.astype(np.float32), allow_pickle=False)
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _embed_keys(
    embedder: Embedder,
    cache: EmbeddingCache,
    jobs: list[tuple[str, Callable[[], Image.Image]]],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Embed (cache_key, loader) jobs, using the cache. Returns (vectors, errors)."""
    vectors: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    pending: list[tuple[str, Callable[[], Image.Image]]] = []
    for key, loader in jobs:
        if key in vectors or key in errors:
            continue
        hit = cache.get(key)
        if hit is not None:
            vectors[key] = hit
        else:
            pending.append((key, loader))
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        keys, images = [], []
        for key, loader in chunk:
            try:
                images.append(loader())
                keys.append(key)
            except Exception as exc:
                errors[key] = f"decode failed: {exc}"
        if not images:
            continue
        vecs = embedder.embed(images)
        for key, vec in zip(keys, vecs, strict=True):
            cache.put(key, vec)
            vectors[key] = vec
    return vectors, errors


def embed_candidates(
    embedder: Embedder, cache: EmbeddingCache, files: list[ImageFile], batch_size: int = 16
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ({sha256: vector}, {sha256: error})."""
    jobs = [(f.sha256, lambda p=f.path: open_rgb(p)) for f in files]
    return _embed_keys(embedder, cache, jobs, batch_size)


def embed_references(
    embedder: Embedder, cache: EmbeddingCache, files: list[ImageFile], flip: bool = True, batch_size: int = 16
) -> np.ndarray:
    """Return an (M, D) matrix; each row is the reference (optionally averaged with its flip)."""
    jobs: list[tuple[str, Callable[[], Image.Image]]] = []
    for f in files:
        jobs.append((f.sha256, lambda p=f.path: open_rgb(p)))
        if flip:
            jobs.append((f.sha256 + ".flip", lambda p=f.path: ImageOps.mirror(open_rgb(p))))
    vectors, errors = _embed_keys(embedder, cache, jobs, batch_size)
    if errors:
        raise ValueError(f"failed to embed references: {errors}")
    rows = []
    for f in files:
        v = vectors[f.sha256]
        if flip:
            v = v + vectors[f.sha256 + ".flip"]
        rows.append(v)
    return l2_normalize(np.stack(rows))
