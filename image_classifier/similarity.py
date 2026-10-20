"""Cosine similarity and the ACCEPT / REJECT / BORDERLINE banding rule (plan §4 step 4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import BandConfig

# Float slack so that e.g. 0.75 - 0.70 still satisfies margin=0.05.
EPS = 1e-9


class Band(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BORDERLINE = "BORDERLINE"


@dataclass(frozen=True)
class BandDecision:
    band: Band
    best: int  # index of r1
    s1: float
    s2: float


def cosine_matrix(candidates: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """(N, D) x (M, D) -> (N, M). Rows are assumed L2-normalized."""
    if candidates.size == 0:
        return np.zeros((0, refs.shape[0]), dtype=np.float32)
    return candidates @ refs.T


def band_decision(scores: np.ndarray, cfg: BandConfig) -> BandDecision:
    order = np.argsort(-scores, kind="stable")
    best = int(order[0])
    s1 = float(scores[best])
    s2 = float(scores[order[1]]) if scores.shape[0] > 1 else 0.0
    if s1 >= cfg.t_high - EPS and (s1 - s2) >= cfg.margin - EPS:
        band = Band.ACCEPT
    elif s1 < cfg.t_low - EPS:
        band = Band.REJECT
    else:
        band = Band.BORDERLINE
    return BandDecision(band, best, s1, s2)
