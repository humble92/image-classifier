"""Output directory management and collision-safe copying. Inputs are never moved."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

MANIFEST = ".image-classifier.json"
REPORT_NAME = "report.csv"
UNMATCHED = "_unmatched"
REVIEW = "_review"


class OutputExistsError(FileExistsError):
    pass


def _owned_entries(out: Path) -> list[str]:
    try:
        data = json.loads((out / MANIFEST).read_text())
        entries = data.get("created", [])
    except (OSError, ValueError, AttributeError):
        return []
    # Only plain direct children; never follow anything that looks like a path.
    return [e for e in entries if isinstance(e, str) and e and "/" not in e and "\\" not in e and e not in (".", "..")]


def prepare_out(out: Path, folders: list[str], overwrite: bool) -> None:
    """Create ``out`` and its folders. With ``overwrite``, remove only what a previous run created."""
    if out.exists():
        if not overwrite:
            raise OutputExistsError(f"output directory exists: {out} (pass --overwrite)")
        if not out.is_dir():
            raise OutputExistsError(f"output path is not a directory: {out}")
        for name in _owned_entries(out):
            target = out / name
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        (out / MANIFEST).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    created = []
    for name in folders:
        d = out / name
        if not d.exists():
            d.mkdir()
            created.append(name)
    created.append(REPORT_NAME)
    (out / MANIFEST).write_text(json.dumps({"created": created}, indent=2))


def copy_unique(src: Path, dest_dir: Path, sha256: str) -> Path:
    """Copy ``src`` into ``dest_dir``; on a name collision append ``_<shortHash>`` (then a counter)."""
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}_{sha256[:8]}{src.suffix}"
        n = 2
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{sha256[:8]}_{n}{src.suffix}"
            n += 1
    shutil.copy2(src, dest)
    return dest
