from pathlib import Path

import pytest

from image_classifier.config import Config, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_matches_defaults():
    assert load_config(ROOT / "classifier.example.toml") == Config()


def test_overrides_and_none_ignored(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[band]\nt_high = 0.8\n")
    cfg = load_config(p, {"band.t_low": 0.5, "band.margin": None, "jev.enabled": False})
    assert (cfg.band.t_high, cfg.band.t_low, cfg.band.margin, cfg.jev.enabled) == (0.8, 0.5, 0.05, False)


@pytest.mark.parametrize(
    "text",
    ["[band]\nt_hgih = 0.8\n", "[nope]\nx = 1\n", "band = 3\n", "[band]\nt_low = 0.9\n", "[jev]\nmax_side = 2000\n"],
)
def test_invalid_config(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_config(p)
