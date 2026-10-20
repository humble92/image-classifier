import base64
import json
import re

import pytest

from image_classifier.jev_judge import (
    BASE_OVERHEAD_BYTES,
    JevItem,
    JevRef,
    build_payload,
    pack_batches,
)

REFS = [JevRef("ref_1", "red_bike", b"R1" * 50), JevRef("ref_2", "blue_bike", b"R2" * 50)]
ITEMS = [JevItem(f"sha{i}", bytes([i]) * (100 + i)) for i in range(5)]
PH = re.compile(r"<img:[^>]+>")


def _uses(obj) -> list[str]:
    return PH.findall(json.dumps(obj))


def _get_path(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def test_every_asset_referenced_exactly_once():
    payload, _ = build_payload("jev-1.14", REFS, ITEMS)
    body = {k: v for k, v in payload.items() if k != "assets"}
    uses = _uses(body)
    assert sorted(uses) == sorted(payload["assets"])  # no unreferenced asset, no duplicate use
    assert len(uses) == len(set(uses)) == len(REFS) + len(ITEMS)


def test_refs_only_in_state_candidates_only_in_own_question():
    payload, qmap = build_payload("jev-1.14", REFS, ITEMS)
    state_uses = set(_uses(payload["state"]))
    assert state_uses == {"<img:ref_1>", "<img:ref_2>"}
    for qid, q in payload["questions"].items():
        assert _uses(q) == [f"<img:{qid}>"]
        assert f"<img:{qid}>" in q["instructions"]
    assert list(qmap.values()) == [i.item_id for i in ITEMS]


def test_criteria_paths_resolve_in_state():
    payload, _ = build_payload("jev-1.14", REFS, ITEMS)
    for q in payload["questions"].values():
        assert q["type"] == "choice"
        assert set(q["criteria"]) == {"ref_1", "ref_2", "none"}
        for key, text in q["criteria"].items():
            for path in re.findall(r"`([^`]+)`", text):
                assert _get_path(payload["state"], path) == f"<img:{key}>"


def test_asset_encoding_roundtrip():
    payload, _ = build_payload("jev-1.14", REFS, ITEMS[:1])
    a = payload["assets"]["<img:c_0>"]
    assert a["type"] == "image" and a["format"] == "jpeg"
    assert base64.b64decode(a["data_base64"]) == ITEMS[0].jpeg


def test_build_requires_refs_and_items():
    with pytest.raises(ValueError):
        build_payload("m", [], ITEMS)
    with pytest.raises(ValueError):
        build_payload("m", REFS, [])


def test_pack_respects_max_batch():
    items = [JevItem(str(i), b"x" * 10) for i in range(37)]
    batches, oversize = pack_batches(items, REFS, max_batch=16, max_request_bytes=10**7)
    assert [len(b) for b in batches] == [16, 16, 5]
    assert not oversize
    assert [i for b in batches for i in b] == items  # order preserved, nothing lost


def test_pack_respects_byte_limit_and_real_payload_fits():
    items = [JevItem(str(i), b"x" * 30_000) for i in range(10)]
    limit = 150_000
    batches, oversize = pack_batches(items, REFS, max_batch=16, max_request_bytes=limit)
    assert not oversize and len(batches) > 1
    for b in batches:
        payload, _ = build_payload("jev-1.14", REFS, b)
        assert len(json.dumps(payload, separators=(",", ":")).encode()) <= limit


def test_pack_flags_items_that_never_fit():
    big = JevItem("big", b"x" * 200_000)
    small = JevItem("small", b"x" * 10)
    batches, oversize = pack_batches([big, small], REFS, max_batch=16, max_request_bytes=BASE_OVERHEAD_BYTES + 50_000)
    assert oversize == [big]
    assert batches == [[small]]
