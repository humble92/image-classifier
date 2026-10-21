import numpy as np
import pytest

from image_classifier.config import BandConfig
from image_classifier.similarity import Band, band_decision, cosine_matrix

CFG = BandConfig(t_high=0.70, t_low=0.45, margin=0.05)


@pytest.mark.parametrize(
    "scores, band, best",
    [
        ([0.70, 0.65], Band.ACCEPT, 0),  # exactly t_high and exactly margin (float slack)
        ([0.75, 0.70], Band.ACCEPT, 0),  # 0.75 - 0.70 = 0.04999... in floats
        ([0.70, 0.66], Band.BORDERLINE, 0),  # margin too small
        ([0.6999, 0.1], Band.BORDERLINE, 0),  # just under t_high
        ([0.45, 0.1], Band.BORDERLINE, 0),  # exactly t_low is not REJECT
        ([0.4499, 0.1], Band.REJECT, 0),
        ([0.2, 0.9], Band.ACCEPT, 1),
        ([0.80, 0.80], Band.BORDERLINE, 0),  # high similarity to two refs at once
        ([0.95], Band.ACCEPT, 0),  # M = 1: s2 = 0
        ([0.30], Band.REJECT, 0),
    ],
)
def test_band_boundaries(scores, band, best):
    d = band_decision(np.array(scores), CFG)
    assert d.band is band
    assert d.best == best


def test_single_ref_s2_is_zero():
    d = band_decision(np.array([0.5]), CFG)
    assert d.s2 == 0.0 and d.band is Band.BORDERLINE


def test_margin_with_negative_second_score():
    # M = 2 with negative s2 must still compute s1 - s2 correctly.
    d = band_decision(np.array([0.72, -0.3]), CFG)
    assert d.band is Band.ACCEPT and d.s2 == pytest.approx(-0.3)


def test_cosine_matrix_shapes():
    c = np.eye(3, dtype=np.float32)
    r = np.eye(3, dtype=np.float32)[:2]
    s = cosine_matrix(c, r)
    assert s.shape == (3, 2)
    assert s[0, 0] == 1.0 and s[2, 1] == 0.0
    assert cosine_matrix(np.zeros((0, 3), np.float32), r).shape == (0, 2)
