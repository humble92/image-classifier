"""report.csv and the console summary."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Row:
    file: str
    sha256: str = ""
    scores: list[float] = field(default_factory=list)
    band: str = ""
    jev_choice: str = ""
    jev_probabilities: dict[str, float] | None = None
    jev_batch_id: str = ""
    jev_batch_input_tokens: int | None = None
    final_folder: str = ""
    error: str = ""


def write_report(path: Path, rows: list[Row], n_refs: int) -> None:
    header = [
        "file",
        "sha256",
        *[f"s_ref_{i}" for i in range(1, n_refs + 1)],
        "band",
        "jev_choice",
        "jev_probabilities",
        "jev_batch_id",
        "jev_batch_input_tokens",
        "final_folder",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            scores = [f"{s:.4f}" for s in r.scores] + [""] * (n_refs - len(r.scores))
            w.writerow(
                [
                    r.file,
                    r.sha256,
                    *scores,
                    r.band,
                    r.jev_choice,
                    json.dumps(r.jev_probabilities, sort_keys=True) if r.jev_probabilities else "",
                    r.jev_batch_id,
                    "" if r.jev_batch_input_tokens is None else r.jev_batch_input_tokens,
                    r.final_folder,
                    r.error,
                ]
            )


def summarize(
    rows: list[Row],
    jev_requests: int,
    input_tokens: int,
    usd_per_mtok: float,
    dry_run: bool = False,
    estimated_requests: int | None = None,
) -> str:
    lines = []
    if dry_run:
        bands = Counter(r.band or "error" for r in rows)
        lines.append("Dry run: no files copied, no API calls.")
        for band, n in sorted(bands.items()):
            lines.append(f"  {band:<14} {n}")
        if estimated_requests is not None:
            lines.append(f"Estimated Jev requests: {estimated_requests}")
        return "\n".join(lines)
    folders = Counter(r.final_folder or "(error, not copied)" for r in rows)
    lines.append("Images per folder:")
    for name, n in sorted(folders.items()):
        lines.append(f"  {name:<24} {n}")
    lines.append(f"Jev requests: {jev_requests}")
    lines.append(f"Jev input tokens: {input_tokens}")
    lines.append(f"Estimated Jev cost: ${input_tokens / 1e6 * usd_per_mtok:.4f} (at ${usd_per_mtok}/1M input tokens)")
    return "\n".join(lines)
