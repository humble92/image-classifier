"""TypeSafe Jev judge for borderline candidates (plan §4 step 5).

One request = one batch = K questions, one per candidate. Reference images live in
shared ``state`` once per request; each candidate image is attached only to its own
question's ``instructions``. Every asset is referenced exactly once.

Everything Jev-specific (endpoint, headers, response shape) is isolated here because
TypeSafe marks image input as experimental.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import JevConfig

log = logging.getLogger(__name__)

NONE_CHOICE = "none"
REVIEW = "review"

TASK = (
    "Identity matching. References show specific individual objects/subjects. "
    "Candidates are photos that may or may not show one of those same individuals."
)
QUESTION = (
    "Which reference shows the very same individual object/subject as the candidate photo above? "
    "Same individual means identical identity (e.g. the same specific item, not merely the same "
    "model, brand, breed or category). Ignore differences in angle, lighting, background, crop and scale."
)

# Conservative JSON framing allowances used by the packer.
BASE_OVERHEAD_BYTES = 8 * 1024
PER_IMAGE_OVERHEAD_BYTES = 2 * 1024


@dataclass(frozen=True)
class JevRef:
    key: str  # ref_1 .. ref_M
    name: str
    jpeg: bytes


@dataclass(frozen=True)
class JevItem:
    item_id: str  # caller's id, returned in outcomes
    jpeg: bytes


@dataclass(frozen=True)
class JevAnswer:
    choice: str
    confidence: float | None
    probabilities: dict[str, float]


@dataclass
class JevOutcome:
    decision: str  # a ref key, NONE_CHOICE, or REVIEW
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    batch_id: str | None = None
    batch_input_tokens: int | None = None
    error: str | None = None


@dataclass
class BatchRecord:
    batch_id: str
    size: int
    request_bytes: int
    status: str  # ok | validation_error | transient_error | fatal_error
    input_tokens: int | None = None


class JevError(Exception):
    pass


class JevValidationError(JevError):
    """400/422: something in the request was rejected. Bisect to find it."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class JevTransientError(JevError):
    """429/5xx/network failure that persisted through all retries."""


class JevFatalError(JevError):
    """Auth or configuration failure (401/403/404...). Stop calling Jev for this run."""


def b64_len(n: int) -> int:
    return 4 * math.ceil(n / 3)


def placeholder(key: str) -> str:
    return f"<img:{key}>"


def _asset(jpeg: bytes) -> dict[str, str]:
    return {"type": "image", "format": "jpeg", "data_base64": base64.b64encode(jpeg).decode("ascii")}


def build_payload(model: str, refs: list[JevRef], items: list[JevItem]) -> tuple[dict[str, Any], dict[str, str]]:
    """Build one request. Returns (payload, {question_id: item_id})."""
    if not refs or not items:
        raise ValueError("need at least one reference and one candidate")
    state = {
        "task": TASK,
        "references": {r.key: {"name": r.name, "image": placeholder(r.key)} for r in refs},
    }
    criteria = {r.key: f"Same individual as `references.{r.key}.image`" for r in refs}
    criteria[NONE_CHOICE] = "Not the same individual as any reference (may be similar-looking or the same type)"

    assets = {placeholder(r.key): _asset(r.jpeg) for r in refs}
    questions: dict[str, Any] = {}
    qmap: dict[str, str] = {}
    for i, item in enumerate(items):
        qid = f"c_{i}"
        questions[qid] = {
            "type": "choice",
            "instructions": f"Candidate photo: {placeholder(qid)}\n{QUESTION}",
            "criteria": dict(criteria),
        }
        assets[placeholder(qid)] = _asset(item.jpeg)
        qmap[qid] = item.item_id
    payload = {"model": model, "state": state, "questions": questions, "assets": assets}
    return payload, qmap


def estimate_item_bytes(jpeg_len: int) -> int:
    return b64_len(jpeg_len) + PER_IMAGE_OVERHEAD_BYTES


def pack_batches(
    items: list[JevItem], refs: list[JevRef], max_batch: int, max_request_bytes: int
) -> tuple[list[list[JevItem]], list[JevItem]]:
    """Greedy first-fit in input order. Returns (batches, items_too_large_to_send)."""
    base = BASE_OVERHEAD_BYTES + sum(estimate_item_bytes(len(r.jpeg)) for r in refs)
    batches: list[list[JevItem]] = []
    oversize: list[JevItem] = []
    current: list[JevItem] = []
    current_bytes = base
    for item in items:
        size = estimate_item_bytes(len(item.jpeg))
        if base + size > max_request_bytes:
            oversize.append(item)
            continue
        if current and (len(current) >= max_batch or current_bytes + size > max_request_bytes):
            batches.append(current)
            current, current_bytes = [], base
        current.append(item)
        current_bytes += size
    if current:
        batches.append(current)
    return batches, oversize


def parse_answers(body: Any, qids: list[str], allowed: set[str]) -> tuple[dict[str, JevAnswer | str], int | None]:
    """Parse a response into {qid: JevAnswer or error string} plus usage.input_tokens.

    Assumed shape (verify in the M3 pilot):
    {"answers": {"c_0": {"choice": "ref_1", "confidence": 0.9, "probabilities": {...}}},
     "usage": {"input_tokens": 1234}}
    """
    out: dict[str, JevAnswer | str] = {}
    tokens: int | None = None
    if not isinstance(body, dict):
        return {q: "malformed response: not an object" for q in qids}, None
    usage = body.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
        tokens = usage["input_tokens"]
    answers = body.get("answers")
    if not isinstance(answers, dict):
        return {q: "malformed response: missing answers" for q in qids}, tokens
    for qid in qids:
        a = answers.get(qid)
        if not isinstance(a, dict):
            out[qid] = "missing answer"
            continue
        choice = a.get("choice")
        probs = a.get("probabilities")
        conf = a.get("confidence")
        if choice not in allowed:
            out[qid] = f"invalid choice {choice!r}"
            continue
        if not isinstance(probs, dict) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in probs.values()
        ):
            out[qid] = "missing or malformed probabilities"
            continue
        out[qid] = JevAnswer(
            choice=choice,
            confidence=float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None,
            probabilities={str(k): float(v) for k, v in probs.items()},
        )
    return out, tokens


def decide(answer: JevAnswer, accept: float) -> str:
    p = answer.probabilities.get(answer.choice)
    if p is not None and p >= accept:
        return answer.choice
    return REVIEW


class JevClient:
    """Thin HTTP client: headers, retries with backoff, error classification."""

    def __init__(
        self,
        cfg: JevConfig,
        api_key: str,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise JevFatalError(f"no API key; set {cfg.api_key_env}")
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + cfg.endpoint
        self.http = http or httpx.Client(timeout=cfg.timeout_s)
        self.sleep = sleep
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }
        if cfg.long_context:
            self.headers["TypeSafe-Danger-Zone-Long-Context"] = "true"

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.strip().isdigit():
                return min(float(retry_after), 60.0)
        return min(2.0**attempt, 30.0) + random.uniform(0, 0.5)

    def post(self, body: bytes) -> Any:
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            resp: httpx.Response | None = None
            try:
                resp = self.http.post(self.url, content=body, headers=self.headers)
            except httpx.TransportError as exc:
                last = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise JevTransientError(f"non-JSON 200 response: {exc}") from exc
                if resp.status_code in (400, 422):
                    raise JevValidationError(resp.status_code, resp.text)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                else:
                    raise JevFatalError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            if attempt < self.cfg.max_retries:
                delay = self._backoff(attempt, resp)
                log.warning("Jev %s; retrying in %.1fs", last, delay)
                self.sleep(delay)
        raise JevTransientError(f"gave up after {self.cfg.max_retries + 1} attempts: {last}")


@dataclass
class JevJudge:
    client: JevClient
    refs: list[JevRef]
    cfg: JevConfig
    requests_made: int = 0
    input_tokens: int = 0
    batches: list[BatchRecord] = field(default_factory=list)
    disabled_reason: str | None = None

    @property
    def allowed_choices(self) -> set[str]:
        return {r.key for r in self.refs} | {NONE_CHOICE}

    def run(self, items: list[JevItem]) -> dict[str, JevOutcome]:
        out: dict[str, JevOutcome] = {}
        max_bytes = int(self.cfg.max_request_mib * 1024 * 1024)
        batches, oversize = pack_batches(items, self.refs, self.cfg.max_batch, max_bytes)
        for item in oversize:
            out[item.item_id] = JevOutcome(REVIEW, error="too large for one Jev request with references")
        for batch in batches:
            self._judge(batch, out)
        return out

    def _review_all(self, batch: list[JevItem], out: dict[str, JevOutcome], error: str, batch_id: str | None = None):
        for item in batch:
            out[item.item_id] = JevOutcome(REVIEW, batch_id=batch_id, error=error)

    def _judge(self, batch: list[JevItem], out: dict[str, JevOutcome]) -> None:
        if self.disabled_reason:
            self._review_all(batch, out, f"jev disabled: {self.disabled_reason}")
            return
        if self.requests_made >= self.cfg.max_requests:
            self._review_all(batch, out, "jev request cap reached")
            return

        payload, qmap = build_payload(self.cfg.model, self.refs, batch)
        body = json.dumps(payload, separators=(",", ":")).encode()
        batch_id = f"b{len(self.batches) + 1:04d}"
        record = BatchRecord(batch_id, len(batch), len(body), "ok")
        self.batches.append(record)
        self.requests_made += 1
        try:
            response = self.client.post(body)
        except JevValidationError as exc:
            record.status = "validation_error"
            log.error("Jev batch %s rejected (%s): %s", batch_id, exc.status, exc.body[:2000])
            if len(batch) == 1:
                self._review_all(batch, out, f"jev rejected: {exc}", batch_id)
                return
            mid = len(batch) // 2
            self._judge(batch[:mid], out)
            self._judge(batch[mid:], out)
            return
        except JevTransientError as exc:
            record.status = "transient_error"
            self._review_all(batch, out, f"jev unavailable: {exc}", batch_id)
            return
        except JevFatalError as exc:
            record.status = "fatal_error"
            self.disabled_reason = str(exc)
            self._review_all(batch, out, f"jev fatal: {exc}", batch_id)
            return

        answers, tokens = parse_answers(response, list(qmap), self.allowed_choices)
        record.input_tokens = tokens
        if tokens is not None:
            self.input_tokens += tokens
        for qid, item_id in qmap.items():
            a = answers[qid]
            if isinstance(a, str):
                out[item_id] = JevOutcome(REVIEW, batch_id=batch_id, batch_input_tokens=tokens, error=a)
            else:
                out[item_id] = JevOutcome(
                    decide(a, self.cfg.accept),
                    choice=a.choice,
                    probabilities=a.probabilities,
                    batch_id=batch_id,
                    batch_input_tokens=tokens,
                )
