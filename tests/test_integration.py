"""End-to-end on the synthetic fixture set with Jev mocked."""

import base64
import csv
import json
import shutil
from pathlib import Path

import httpx
from typer.testing import CliRunner

from image_classifier import cli
from image_classifier.jev_judge import JevClient
from image_classifier.pipeline import classify
from image_classifier.preprocess import jev_jpeg

BORDERLINE = "BORDERLINE"


def _report(out: Path) -> dict[str, dict]:
    with open(out / "report.csv", newline="") as fh:
        return {Path(r["file"]).name: r for r in csv.DictReader(fh)}


def _folder_contents(out: Path) -> dict[str, set[str]]:
    return {d.name: {p.name for p in d.iterdir()} for d in out.iterdir() if d.is_dir()}


def test_no_jev_routes_and_reports(dataset, cfg, fake_embedder):
    cfg.jev.enabled = False
    before = {p: p.read_bytes() for p in dataset["input"].rglob("*") if p.is_file()}
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder)
    assert [(r.key, r.name) for r in res.refs] == [("ref_1", "blue_bike"), ("ref_2", "red_bike")]

    rep = _report(dataset["out"])
    assert len(rep) == len(dataset["expected"]) + 1  # + broken.jpg
    for path, expected in dataset["expected"].items():
        row = rep[Path(path).name]
        want = "_review" if expected == BORDERLINE else expected
        assert row["final_folder"] == want, (path, row)
        if expected == BORDERLINE:
            assert row["band"] == "BORDERLINE"
    assert rep["copy_of_ref.png"]["band"] == "REF_DUPLICATE"
    assert rep["broken.jpg"]["final_folder"] == "" and rep["broken.jpg"]["error"].startswith("unreadable")

    contents = _folder_contents(dataset["out"])
    assert set(contents) == {"blue_bike", "red_bike", "_unmatched", "_review"}
    assert sum(len(v) for v in contents.values()) == len(dataset["expected"])  # each image exactly once
    assert len(contents["_review"]) == 5
    # Inputs untouched.
    assert {p: p.read_bytes() for p in dataset["input"].rglob("*") if p.is_file()} == before
    assert res.jev_requests == 0


def _mock_jev(dataset, answers_by_name):
    """MockTransport that answers by recognising each candidate's exact Jev JPEG."""
    by_b64 = {
        base64.b64encode(jev_jpeg(dataset["input"] / name)).decode(): ans for name, ans in answers_by_name.items()
    }
    requests = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        requests.append(body)
        answers = {}
        for qid in body["questions"]:
            choice, p = by_b64[body["assets"][f"<img:{qid}>"]["data_base64"]]
            rest = (1 - p) / 2
            probs = {k: (p if k == choice else rest) for k in ("ref_1", "ref_2", "none")}
            answers[qid] = {"choice": choice, "confidence": p, "probabilities": probs}
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 1000 * len(answers)}})

    return handler, requests


def test_with_mocked_jev(dataset, cfg, fake_embedder):
    handler, requests = _mock_jev(
        dataset,
        {
            "look_a0.png": ("ref_2", 0.95),
            "look_a1.png": ("ref_2", 0.90),
            "look_a2.png": ("none", 0.90),
            "look_a3.png": ("ref_2", 0.60),  # below jev.accept -> review
            "ab_mix.png": ("ref_1", 0.85),
        },
    )
    client = JevClient(cfg.jev, "k", http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, jev_client=client)
    assert res.jev_requests == 1 and len(requests) == 1
    assert len(requests[0]["questions"]) == 5
    # References carried once in shared state; only the batch's images are assets.
    assert len(requests[0]["assets"]) == 2 + 5
    assert res.input_tokens == 5000

    rep = _report(dataset["out"])
    assert rep["look_a0.png"]["final_folder"] == "red_bike"
    assert rep["look_a1.png"]["final_folder"] == "red_bike"
    assert rep["look_a2.png"]["final_folder"] == "_unmatched"
    assert rep["look_a3.png"]["final_folder"] == "_review"
    assert rep["ab_mix.png"]["final_folder"] == "blue_bike"
    assert rep["look_a3.png"]["jev_choice"] == "ref_2"
    assert json.loads(rep["look_a0.png"]["jev_probabilities"])["ref_2"] == 0.95
    assert rep["look_a0.png"]["jev_batch_id"] == "b0001"
    assert rep["look_a0.png"]["jev_batch_input_tokens"] == "5000"


def test_duplicate_borderline_content_judged_once(dataset, cfg, fake_embedder):
    shutil.copy(dataset["input"] / "look_a0.png", dataset["input"] / "sub" / "look_a0.png")
    names = ["look_a0.png", "look_a1.png", "look_a2.png", "look_a3.png", "ab_mix.png"]
    handler, requests = _mock_jev(dataset, {n: ("none", 0.9) for n in names})
    client = JevClient(cfg.jev, "k", http=httpx.Client(transport=httpx.MockTransport(handler)))
    classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, jev_client=client)
    assert len(requests[0]["questions"]) == 5
    contents = _folder_contents(dataset["out"])
    # Both copies routed; the second gets a hash suffix.
    look = sorted(n for n in contents["_unmatched"] if n.startswith("look_a0"))
    assert len(look) == 2 and look[0] == "look_a0.png" and look[1].startswith("look_a0_")


def test_dry_run_estimates_without_side_effects(dataset, cfg, fake_embedder):
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, dry_run=True)
    assert not dataset["out"].exists()
    assert res.estimated_requests == 1 and res.jev_requests == 0
    cfg.jev.max_batch = 2
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, dry_run=True)
    assert res.estimated_requests == 3
    cfg.jev.max_requests = 1
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, dry_run=True)
    assert res.estimated_requests == 1 and any("max-jev-requests" in n for n in res.notes)


def test_missing_api_key_falls_back_to_review(dataset, cfg, fake_embedder, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    res = classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder)
    assert any("Jev unavailable" in n for n in res.notes)
    assert len(_folder_contents(dataset["out"])["_review"]) == 5


def test_rerun_uses_cache(dataset, cfg, fake_embedder):
    cfg.jev.enabled = False
    classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder)
    seen = fake_embedder.images_seen
    cfg.band.t_high = 0.95  # threshold change needs no recomputation
    classify(dataset["input"], [dataset["refs"]], dataset["out"], cfg, fake_embedder, overwrite=True)
    assert fake_embedder.images_seen == seen


def test_cli_classify(dataset, cfg, fake_embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "embedder_factory", lambda c: fake_embedder)
    conf = tmp_path / "c.toml"
    conf.write_text(
        f'[embed]\nflip_refs = false\ncache_dir = "{Path(cfg.embed.cache_dir).as_posix()}"\n[band]\nt_high = 0.9\nt_low = 0.4\n'
    )
    runner = CliRunner()
    args = ["classify", str(dataset["input"]), "--refs", str(dataset["refs"]), "--out", str(dataset["out"])]
    args += ["--config", str(conf), "--no-jev"]
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "Jev requests: 0" in r.output and "_review" in r.output

    r = runner.invoke(cli.app, args)
    assert r.exit_code == 2 and "exists" in r.output

    r = runner.invoke(cli.app, [*args, "--overwrite", "--dry-run"])
    assert r.exit_code == 0 and "Dry run" in r.output

    r = runner.invoke(cli.app, [*args, "--overwrite", "--t-low", "0.95"])
    assert r.exit_code == 2  # t_low > t_high rejected by config validation


def test_cli_refs_as_files(dataset, cfg, fake_embedder, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "embedder_factory", lambda c: fake_embedder)
    conf = tmp_path / "c.toml"
    conf.write_text(f'[embed]\nflip_refs = false\ncache_dir = "{Path(cfg.embed.cache_dir).as_posix()}"\n')
    r = CliRunner().invoke(
        cli.app,
        [
            "classify",
            str(dataset["input"]),
            "--refs",
            str(dataset["refs"] / "red bike.png"),
            "--out",
            str(dataset["out"]),
            "--config",
            str(conf),
            "--no-jev",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "ref_1=red_bike" in r.output
