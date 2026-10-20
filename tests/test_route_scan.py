import json

import pytest
from PIL import Image

from image_classifier.io_scan import load_refs, sanitize_name, scan_inputs
from image_classifier.route import MANIFEST, OutputExistsError, copy_unique, prepare_out


def _img(path, color=(1, 2, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def test_copy_unique_collisions(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    a = _img(tmp_path / "x" / "photo.png", (1, 1, 1))
    b = _img(tmp_path / "y" / "photo.png", (2, 2, 2))
    c = _img(tmp_path / "z" / "photo.png", (3, 3, 3))
    p1 = copy_unique(a, dest, "aaaaaaaa1111")
    p2 = copy_unique(b, dest, "bbbbbbbb2222")
    p3 = copy_unique(c, dest, "bbbbbbbb2222")  # same short hash as p2
    assert [p.name for p in (p1, p2, p3)] == ["photo.png", "photo_bbbbbbbb.png", "photo_bbbbbbbb_2.png"]
    assert a.exists()  # copy, never move


def test_prepare_out_refuses_existing_without_overwrite(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(OutputExistsError):
        prepare_out(out, ["a"], overwrite=False)


def test_overwrite_clears_only_tool_created(tmp_path):
    out = tmp_path / "out"
    prepare_out(out, ["a", "_review"], overwrite=False)
    (out / "a" / "old.png").write_bytes(b"x")
    (out / "report.csv").write_text("old")
    (out / "user_notes.txt").write_text("keep me")
    (out / "keep_dir").mkdir()
    prepare_out(out, ["b", "_review"], overwrite=True)
    assert not (out / "a").exists() and not (out / "report.csv").exists()
    assert (out / "user_notes.txt").read_text() == "keep me"
    assert (out / "keep_dir").is_dir() and (out / "b").is_dir()


def test_overwrite_ignores_malicious_manifest(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / MANIFEST).write_text(json.dumps({"created": ["../victim", "/etc", "..", 5]}))
    prepare_out(out, ["a"], overwrite=True)
    assert victim.is_dir()


def test_scan_reports_unreadable_and_excludes_out(tmp_path):
    inp = tmp_path / "in"
    _img(inp / "ok.png")
    _img(inp / "deep" / "ok2.JPG")
    (inp / "bad.png").write_bytes(b"nope")
    (inp / "skip.txt").write_text("x")
    _img(inp / "out" / "already_sorted.png")
    files, errors = scan_inputs(inp, recursive=True, exclude=inp / "out")
    assert sorted(f.path.name for f in files) == ["ok.png", "ok2.JPG"]
    assert [e.path.name for e in errors] == ["bad.png"]
    files, _ = scan_inputs(inp, recursive=False)
    assert [f.path.name for f in files] == ["ok.png"]


def test_truncated_image_is_error(tmp_path):
    p = _img(tmp_path / "in" / "t.png")
    p.write_bytes(p.read_bytes()[:60])
    _, errors = scan_inputs(tmp_path / "in")
    assert len(errors) == 1


@pytest.mark.parametrize(
    "stem, name",
    [("red bike", "red_bike"), ("_review", "review"), ("../../etc", "etc"), ("***", "ref"), ("a" * 100, "a" * 64)],
)
def test_sanitize(stem, name):
    assert sanitize_name(stem) == name


def test_load_refs_counts_names_and_dupes(tmp_path):
    d = tmp_path / "refs"
    _img(d / "Cat.png", (1, 0, 0))
    _img(d / "cat.jpg", (0, 1, 0))
    _img(d / "dog!.png", (0, 0, 1))
    refs = load_refs([d])
    assert [r.key for r in refs] == ["ref_1", "ref_2", "ref_3"]
    assert [r.name for r in refs] == ["Cat", "cat_2", "dog"]

    for i in range(3):
        _img(d / f"more{i}.png", (i, 9, 9))
    with pytest.raises(ValueError, match="1-5"):
        load_refs([d])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="1-5"):
        load_refs([tmp_path / "empty"])

    same = tmp_path / "same"
    _img(same / "a.png")
    _img(same / "b.png")
    with pytest.raises(ValueError, match="identical"):
        load_refs([same])
    with pytest.raises(FileNotFoundError):
        load_refs([tmp_path / "missing.png"])
