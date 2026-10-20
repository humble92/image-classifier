"""Threshold sweep on a labeled set (plan §6).

Labeled layout: ``LABELED_DIR/<ref_name>/*.jpg`` for true matches of each reference and
``LABELED_DIR/none/*.jpg`` for non-matches (include hard negatives: same type, different
individual). ``<ref_name>`` is the sanitized reference folder name the classifier uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BandConfig
from .io_scan import Reference, scan_inputs

NONE_LABELS = {"none", "_none", "_unmatched"}


@dataclass(frozen=True)
class SweepResult:
    t_high: float
    t_low: float
    margin: float
    accept_precision: float  # nan when nothing is accepted
    accepted: int
    reject_recall_loss: float  # share of true matches sent to REJECT
    borderline_fraction: float


def load_labeled(labeled_dir: Path, refs: list[Reference]):
    """Return (files, labels) with label = ref index or -1 for none."""
    by_name = {r.name: i for i, r in enumerate(refs)}
    files, labels = [], []
    unknown = []
    for sub in sorted(p for p in labeled_dir.iterdir() if p.is_dir()):
        if sub.name in NONE_LABELS:
            label = -1
        elif sub.name in by_name:
            label = by_name[sub.name]
        else:
            unknown.append(sub.name)
            continue
        found, _ = scan_inputs(sub, recursive=True)
        files.extend(found)
        labels.extend([label] * len(found))
    if unknown:
        raise ValueError(f"label folders {unknown} match no reference; references are {sorted(by_name)}")
    return files, np.asarray(labels, dtype=int)


def evaluate(scores: np.ndarray, labels: np.ndarray, band: BandConfig) -> SweepResult:
    n, m = scores.shape
    if n == 0:
        raise ValueError("no labeled images")
    order = np.argsort(-scores, axis=1, kind="stable")
    best = order[:, 0]
    s1 = scores[np.arange(n), best]
    s2 = scores[np.arange(n), order[:, 1]] if m > 1 else np.zeros(n)
    eps = 1e-9
    accept = (s1 >= band.t_high - eps) & ((s1 - s2) >= band.margin - eps)
    reject = ~accept & (s1 < band.t_low - eps)
    border = ~accept & ~reject
    correct = accept & (best == labels)
    n_acc = int(accept.sum())
    positives = labels >= 0
    return SweepResult(
        t_high=band.t_high,
        t_low=band.t_low,
        margin=band.margin,
        accept_precision=float(correct.sum() / n_acc) if n_acc else float("nan"),
        accepted=n_acc,
        reject_recall_loss=float((reject & positives).sum() / positives.sum()) if positives.any() else 0.0,
        borderline_fraction=float(border.mean()),
    )


def sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    t_values: np.ndarray | None = None,
    margins: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08, 0.10),
) -> list[SweepResult]:
    if t_values is None:
        t_values = np.round(np.arange(0.20, 0.96, 0.025), 3)
    results = []
    for t_high in t_values:
        for t_low in t_values[t_values <= t_high]:
            for margin in margins:
                results.append(evaluate(scores, labels, BandConfig(float(t_high), float(t_low), margin)))
    return results


def choose(
    results: list[SweepResult], min_precision: float = 0.98, max_recall_loss: float = 0.02
) -> list[SweepResult]:
    """Configs meeting both targets, cheapest (least borderline) first."""
    ok = [
        r
        for r in results
        if r.accepted > 0 and r.accept_precision >= min_precision and r.reject_recall_loss <= max_recall_loss
    ]
    return sorted(ok, key=lambda r: (r.borderline_fraction, r.reject_recall_loss, -r.accepted))


def format_results(rows: list[SweepResult], limit: int = 10) -> str:
    lines = [f"{'t_high':>7} {'t_low':>7} {'margin':>7} {'acc_prec':>9} {'accepted':>9} {'rej_loss':>9} {'border':>7}"]
    for r in rows[:limit]:
        lines.append(
            f"{r.t_high:7.3f} {r.t_low:7.3f} {r.margin:7.3f} {r.accept_precision:9.3f} "
            f"{r.accepted:9d} {r.reject_recall_loss:9.3f} {r.borderline_fraction:7.3f}"
        )
    return "\n".join(lines)
