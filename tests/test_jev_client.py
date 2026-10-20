"""Contract tests: JevClient/JevJudge against recorded response fixtures via httpx.MockTransport."""

import base64
import json
from pathlib import Path

import httpx
import pytest

from image_classifier.config import JevConfig
from image_classifier.jev_judge import (
    REVIEW,
    JevClient,
    JevFatalError,
    JevItem,
    JevJudge,
    JevRef,
    JevTransientError,
    parse_answers,
)

FIX = Path(__file__).parent / "fixtures" / "jev"
REFS = [JevRef("ref_1", "red", b"R1"), JevRef("ref_2", "blue", b"R2")]


def fixture(name):
    return json.loads((FIX / name).read_text())


def make(handler, **cfg_kw):
    cfg = JevConfig(**cfg_kw)
    sleeps = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = JevClient(cfg, "test-key", http=http, sleep=sleeps.append)
    return JevJudge(client, REFS, cfg), sleeps


def items(n):
    return [JevItem(f"sha{i}", bytes([65 + i]) * 10) for i in range(n)]


def test_success_and_headers():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, json=fixture("success.json"))

    judge, _ = make(handler, accept=0.80)
    out = judge.run(items(3))
    req = seen[0]
    assert req.url.path == "/v1/systemone"
    assert req.headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert req.headers["Authorization"] == "Bearer test-key"
    assert "TypeSafe-Danger-Zone-Long-Context" not in req.headers
    body = json.loads(req.content)
    assert set(body["questions"]) == {"c_0", "c_1", "c_2"} and body["model"] == "jev-1.14"

    assert out["sha0"].decision == "ref_1" and out["sha0"].batch_id == "b0001"
    assert out["sha1"].decision == "none"
    assert out["sha2"].decision == REVIEW  # 0.55 < accept
    assert out["sha2"].choice == "ref_2"
    assert judge.requests_made == 1 and judge.input_tokens == 4321
    assert out["sha0"].batch_input_tokens == 4321


def test_long_context_header():
    seen = []
    judge, _ = make(lambda r: seen.append(r) or httpx.Response(200, json=fixture("success.json")), long_context=True)
    judge.run(items(1))
    assert seen[0].headers["TypeSafe-Danger-Zone-Long-Context"] == "true"


def test_422_bisects_to_bad_image():
    bad = b"Z" * 10
    bad_b64 = base64.b64encode(bad).decode()
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(len(body["questions"]))
        if any(a["data_base64"] == bad_b64 for a in body["assets"].values()):
            return httpx.Response(422, json=fixture("error_422.json"))
        answers = {
            q: {"choice": "none", "confidence": 0.9, "probabilities": {"ref_1": 0.05, "ref_2": 0.05, "none": 0.9}}
            for q in body["questions"]
        }
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 10}})

    its = items(4)
    its[2] = JevItem("bad", bad)
    judge, _ = make(handler)
    out = judge.run(its)
    assert out["bad"].decision == REVIEW and "rejected" in out["bad"].error
    assert all(out[i.item_id].decision == "none" for i in its if i.item_id != "bad")
    assert calls == [4, 2, 2, 1, 1]  # full, ok half, bad half, then its two singles
    assert judge.requests_made == 5


def test_429_retries_then_succeeds():
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(503),
            httpx.Response(200, json=fixture("success.json")),
        ]
    )
    judge, sleeps = make(lambda r: next(responses), max_retries=3)
    out = judge.run(items(3))
    assert out["sha0"].decision == "ref_1"
    assert sleeps[0] == 3.0 and len(sleeps) == 2
    assert judge.requests_made == 1  # retries are one logical request


def test_persistent_5xx_routes_batch_to_review():
    judge, sleeps = make(lambda r: httpx.Response(500), max_retries=3)
    out = judge.run(items(2))
    assert len(sleeps) == 3
    assert all(o.decision == REVIEW and "unavailable" in o.error for o in out.values())


def test_network_error_is_transient():
    def handler(r):
        raise httpx.ConnectError("boom")

    judge, _ = make(handler, max_retries=1)
    out = judge.run(items(1))
    assert out["sha0"].decision == REVIEW


def test_auth_failure_disables_jev_for_rest_of_run():
    calls = []

    def handler(r):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    judge, _ = make(handler, max_batch=1)
    out = judge.run(items(3))
    assert len(calls) == 1
    assert all(o.decision == REVIEW for o in out.values())
    assert judge.disabled_reason


def test_malformed_and_missing_answers_go_to_review():
    judge, _ = make(lambda r: httpx.Response(200, json=fixture("malformed.json")))
    out = judge.run(items(4))
    assert "invalid choice" in out["sha0"].error
    assert "probabilities" in out["sha1"].error
    assert out["sha2"].error == "missing answer"
    assert out["sha3"].error == "missing answer"  # c_3 absent
    assert all(o.decision == REVIEW for o in out.values())


def test_non_json_200_is_transient():
    judge, _ = make(lambda r: httpx.Response(200, text="<html>"), max_retries=0)
    assert judge.run(items(1))["sha0"].decision == REVIEW


def test_request_cap():
    judge, _ = make(lambda r: httpx.Response(200, json=fixture("success.json")), max_batch=1, max_requests=2)
    out = judge.run(items(3))
    assert judge.requests_made == 2
    assert out["sha2"].error == "jev request cap reached"


def test_missing_api_key():
    with pytest.raises(JevFatalError):
        JevClient(JevConfig(), "")


def test_parse_answers_rejects_bool_probabilities():
    body = {"answers": {"c_0": {"choice": "none", "probabilities": {"none": True}}}}
    parsed, tokens = parse_answers(body, ["c_0"], {"ref_1", "none"})
    assert isinstance(parsed["c_0"], str) and tokens is None


def test_transient_error_type():
    cfg = JevConfig(max_retries=0)
    client = JevClient(cfg, "k", http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(502))))
    with pytest.raises(JevTransientError):
        client.post(b"{}")
