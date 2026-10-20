import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from conftest import pattern, render
from image_classifier import calibration, cli
from image_classifier.config import BandConfig


def test_evaluate_metrics():
    scores = np.array(
        [
            [0.90, 0.10],  # label 0, accept correct
            [0.80, 0.78],  # label 1, borderline (margin)
            [0.20, 0.30],  # label 1, reject -> recall loss
            [0.75, 0.10],  # none, accepted -> wrong
            [0.10, 0.05],  # none, reject
        ]
    )
    labels = np.array([0, 1, 1, -1, -1])
    r = calibration.evaluate(scores, labels, BandConfig(0.70, 0.45, 0.05))
    assert r.accepted == 2 and r.accept_precision == 0.5
    assert r.reject_recall_loss == pytest.approx(1 / 3)  # 1 of 3 true matches rejected
    assert r.borderline_fraction == 0.2


def test_sweep_finds_separating_thresholds():
    rng = np.random.default_rng(0)
    pos = np.c_[rng.uniform(0.8, 0.95, 50), rng.uniform(0.0, 0.3, 50)]
    neg = rng.uniform(0.0, 0.5, (50, 2))
    scores, labels = np.r_[pos, neg], np.r_[np.zeros(50, int), -np.ones(50, int)]
    chosen = calibration.choose(calibration.sweep(scores, labels))
    assert chosen
    best = chosen[0]
    assert best.accept_precision == 1.0 and best.borderline_fraction == 0.0
    assert 0.5 <= best.t_high <= 0.8


def test_cli_calibrate(tmp_path, fake_embedder, monkeypatch):
    monkeypatch.setattr(cli, "embedder_factory", lambda c: fake_embedder)
    refs, lab = tmp_path / "refs", tmp_path / "labeled"
    for d in (refs, lab / "alpha", lab / "none"):
        d.mkdir(parents=True)
    A = pattern(1)
    render(A).save(refs / "alpha.png")
    for i in range(5):
        render(A, noise=5, seed=i).save(lab / "alpha" / f"{i}.png")
        render(pattern(50 + i)).save(lab / "none" / f"{i}.png")
    conf = tmp_path / "c.toml"
    conf.write_text(f'[embed]\nflip_refs = false\ncache_dir = "{(tmp_path / "cache").as_posix()}"\n')
    r = CliRunner().invoke(cli.app, ["calibrate", str(lab), "--refs", str(refs), "--config", str(conf)])
    assert r.exit_code == 0, r.output
    assert "Labeled images: 10 (5 matches, 5 none)" in r.output
    assert "[band]" in r.output

    (lab / "mystery").mkdir()
    Image.new("RGB", (10, 10)).save(lab / "mystery" / "x.png")
    r = CliRunner().invoke(cli.app, ["calibrate", str(lab), "--refs", str(refs), "--config", str(conf)])
    assert r.exit_code == 2 and "mystery" in r.output
