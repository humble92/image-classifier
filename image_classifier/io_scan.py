"""Discovery, validation and hashing of candidate and reference images."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:  # HEIC is optional
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    HEIF_AVAILABLE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
MAX_REFS = 5


@dataclass(frozen=True)
class ImageFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class Reference:
    key: str  # ref_1 .. ref_M, stable id used in scores and Jev payloads
    name: str  # sanitized, unique output folder name
    image: ImageFile


@dataclass(frozen=True)
class ScanError:
    path: Path
    error: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_readable(path: Path) -> str | None:
    """Return an error string if Pillow cannot decode the file, else None."""
    if path.suffix.lower() == ".heic" and not HEIF_AVAILABLE:
        return "heic support not installed (pip install pillow-heif)"
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() does not decode pixel data; load() catches truncated files.
        with Image.open(path) as im:
            im.load()
    except Exception as exc:  # Pillow raises many exception types for bad files
        return f"unreadable: {type(exc).__name__}: {exc}"
    return None


def _iter_image_paths(root: Path, recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def scan_inputs(root: Path, recursive: bool = True, exclude: Path | None = None) -> tuple[list[ImageFile], list[ScanError]]:
    """Find candidate images under ``root``. ``exclude`` (e.g. the output dir) is skipped."""
    if not root.is_dir():
        raise NotADirectoryError(f"input directory not found: {root}")
    exclude_resolved = exclude.resolve() if exclude is not None else None
    files: list[ImageFile] = []
    errors: list[ScanError] = []
    for path in _iter_image_paths(root, recursive):
        if exclude_resolved is not None and path.resolve().is_relative_to(exclude_resolved):
            continue
        err = check_readable(path)
        if err:
            errors.append(ScanError(path, err))
            continue
        files.append(ImageFile(path, sha256_file(path)))
    return files, errors


def sanitize_name(stem: str) -> str:
    # Stripping leading "._-" means a name can never be hidden or collide with the
    # reserved "_unmatched" / "_review" folders.
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return name[:64] or "ref"


def load_refs(specs: list[Path]) -> list[Reference]:
    """Resolve reference specs (files and/or directories) into 1..5 references."""
    paths: list[Path] = []
    for spec in specs:
        if spec.is_dir():
            paths.extend(_iter_image_paths(spec, recursive=False))
        elif spec.is_file():
            paths.append(spec)
        else:
            raise FileNotFoundError(f"reference not found: {spec}")
    if not 1 <= len(paths) <= MAX_REFS:
        raise ValueError(f"need 1-{MAX_REFS} reference images, got {len(paths)}")

    refs: list[Reference] = []
    used: set[str] = set()
    seen_sha: dict[str, Path] = {}
    for i, path in enumerate(paths, start=1):
        err = check_readable(path)
        if err:
            raise ValueError(f"reference {path}: {err}")
        sha = sha256_file(path)
        if sha in seen_sha:
            raise ValueError(f"reference {path} is identical to {seen_sha[sha]}")
        seen_sha[sha] = path
        base = sanitize_name(path.stem)
        name, n = base, 2
        while name.lower() in used:  # case-insensitive: macOS/Windows filesystems
            name = f"{base}_{n}"
            n += 1
        used.add(name.lower())
        refs.append(Reference(key=f"ref_{i}", name=name, image=ImageFile(path, sha)))
    return refs
