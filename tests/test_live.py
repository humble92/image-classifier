"""Opt-in: ``TYPESAFE_API_KEY=... pytest -m live``. Sends one real two-question Jev batch.

Set ``TYPESAFE_MODEL`` to try a model id other than the configured default.
"""

import os

import pytest

from conftest import pattern, render
from image_classifier.config import JevConfig
from image_classifier.jev_judge import NONE_CHOICE, JevClient, JevItem, JevJudge, JevRef
from image_classifier.preprocess import jev_jpeg

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set"),
]


def test_one_real_batch(tmp_path):
    A, B = pattern(1), pattern(2)
    paths = {}
    for name, im in {"ref": render(A), "same": render(A, noise=5), "other": render(B)}.items():
        paths[name] = tmp_path / f"{name}.png"
        im.save(paths[name])
    cfg = JevConfig()
    cfg.model = os.environ.get("TYPESAFE_MODEL", cfg.model)
    judge = JevJudge(
        JevClient(cfg, os.environ["TYPESAFE_API_KEY"]), [JevRef("ref_1", "pattern_a", jev_jpeg(paths["ref"]))], cfg
    )
    out = judge.run([JevItem("same", jev_jpeg(paths["same"])), JevItem("other", jev_jpeg(paths["other"]))])
    print({k: vars(v) for k, v in out.items()}, "input_tokens:", judge.input_tokens)
    assert judge.requests_made == 1
    assert all(o.error is None for o in out.values()), out
    assert out["same"].choice == "ref_1"
    assert out["other"].choice == NONE_CHOICE
