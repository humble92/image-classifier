"""End-to-end orchestration: scan -> embed -> band -> (Jev) -> route -> report."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .embed import Embedder, EmbeddingCache, embed_candidates, embed_references
from .io_scan import ImageFile, Reference, load_refs, scan_inputs
from .jev_judge import REVIEW as JEV_REVIEW
from .jev_judge import JevClient, JevError, JevItem, JevJudge, JevRef, pack_batches
from .preprocess import jev_jpeg
from .report import Row, write_report
from .route import REPORT_NAME, REVIEW, UNMATCHED, copy_unique, prepare_out
from .similarity import Band, band_decision, cosine_matrix

log = logging.getLogger(__name__)

REF_DUPLICATE = "REF_DUPLICATE"


@dataclass
class RunResult:
    refs: list[Reference]
    rows: list[Row]
    jev_requests: int = 0
    input_tokens: int = 0
    estimated_requests: int | None = None
    notes: list[str] = field(default_factory=list)


def _folder_for(decision: str, refs_by_key: dict[str, Reference]) -> str:
    if decision in refs_by_key:
        return refs_by_key[decision].name
    if decision == "none":
        return UNMATCHED
    return REVIEW


def classify(
    input_dir: Path,
    ref_specs: list[Path],
    out_dir: Path,
    cfg: Config,
    embedder: Embedder,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    jev_client: JevClient | None = None,
) -> RunResult:
    refs = load_refs(ref_specs)
    refs_by_key = {r.key: r for r in refs}
    ref_by_sha = {r.image.sha256: r for r in refs}
    folders = [r.name for r in refs] + [UNMATCHED, REVIEW]

    # Fail fast on the output dir before doing any expensive work.
    if not dry_run and out_dir.exists() and not overwrite:
        prepare_out(out_dir, folders, overwrite=False)  # raises OutputExistsError

    files, scan_errors = scan_inputs(input_dir, cfg.scan.recursive, exclude=out_dir)
    cache = EmbeddingCache(cfg.embed.cache_dir, embedder.tag)
    R = embed_references(embedder, cache, [r.image for r in refs], cfg.embed.flip_refs, cfg.embed.batch_size)
    # Reference duplicates in the input skip embedding entirely.
    to_embed = [f for f in files if f.sha256 not in ref_by_sha]
    vectors, embed_errors = embed_candidates(embedder, cache, to_embed, cfg.embed.batch_size)

    rows: list[Row] = [Row(file=str(e.path), error=e.error) for e in scan_errors]
    borderline: dict[str, list[tuple[ImageFile, Row]]] = {}  # sha -> files with that content
    routed: list[tuple[ImageFile, Row]] = []

    for f in files:
        row = Row(file=str(f.path), sha256=f.sha256)
        rows.append(row)
        if f.sha256 in ref_by_sha:
            row.band = REF_DUPLICATE
            row.final_folder = ref_by_sha[f.sha256].name
            routed.append((f, row))
            continue
        if f.sha256 in embed_errors:
            row.error = embed_errors[f.sha256]
            continue
        scores = cosine_matrix(vectors[f.sha256][None, :], R)[0]
        row.scores = [float(s) for s in scores]
        d = band_decision(scores, cfg.band)
        row.band = d.band.value
        if d.band is Band.ACCEPT:
            row.final_folder = refs[d.best].name
        elif d.band is Band.REJECT:
            row.final_folder = UNMATCHED
        else:
            borderline.setdefault(f.sha256, []).append((f, row))
            continue
        routed.append((f, row))

    result = RunResult(refs=refs, rows=rows)
    if borderline:
        _judge_borderline(borderline, refs, refs_by_key, cfg, dry_run, jev_client, result)
        routed.extend(pair for pairs in borderline.values() for pair in pairs)

    if dry_run:
        return result

    prepare_out(out_dir, folders, overwrite)
    for f, row in routed:
        copy_unique(f.path, out_dir / row.final_folder, f.sha256)
    write_report(out_dir / REPORT_NAME, rows, len(refs))
    return result


def _judge_borderline(
    borderline: dict[str, list[tuple[ImageFile, Row]]],
    refs: list[Reference],
    refs_by_key: dict[str, Reference],
    cfg: Config,
    dry_run: bool,
    jev_client: JevClient | None,
    result: RunResult,
) -> None:
    def to_review(sha: str, error: str = "") -> None:
        for _, row in borderline[sha]:
            row.final_folder = REVIEW
            row.error = row.error or error

    if not cfg.jev.enabled:
        for sha in borderline:
            to_review(sha)
        return

    jcfg = cfg.jev
    try:
        jev_refs = [JevRef(r.key, r.name, jev_jpeg(r.image.path, jcfg.max_side, jcfg.jpeg_quality)) for r in refs]
    except Exception as exc:
        result.notes.append(f"Jev skipped: reference image could not be prepared: {exc}")
        for sha in borderline:
            to_review(sha, "jev skipped: reference preparation failed")
        return

    items: list[JevItem] = []
    for sha, pairs in borderline.items():
        try:
            items.append(JevItem(sha, jev_jpeg(pairs[0][0].path, jcfg.max_side, jcfg.jpeg_quality)))
        except Exception as exc:
            to_review(sha, f"jev image preparation failed: {exc}")

    if dry_run:
        max_bytes = int(jcfg.max_request_mib * 1024 * 1024)
        batches, oversize = pack_batches(items, jev_refs, jcfg.max_batch, max_bytes)
        result.estimated_requests = min(len(batches), jcfg.max_requests)
        if len(batches) > jcfg.max_requests:
            result.notes.append(
                f"{len(batches)} Jev requests needed but --max-jev-requests is {jcfg.max_requests}; "
                "the remainder would go to _review/"
            )
        if oversize:
            result.notes.append(f"{len(oversize)} candidates too large for any request")
        return

    if jev_client is None:
        try:
            jev_client = JevClient(jcfg, os.environ.get(jcfg.api_key_env, ""))
        except JevError as exc:
            result.notes.append(f"Jev unavailable ({exc}); borderline images routed to _review/")
            for sha in borderline:
                to_review(sha)
            return

    judge = JevJudge(jev_client, jev_refs, jcfg)
    outcomes = judge.run(items)
    result.jev_requests = judge.requests_made
    result.input_tokens = judge.input_tokens
    if judge.disabled_reason:
        result.notes.append(f"Jev disabled mid-run: {judge.disabled_reason}")

    for sha, pairs in borderline.items():
        o = outcomes.get(sha)
        for _, row in pairs:
            if row.final_folder:  # already sent to review during preparation
                continue
            if o is None:
                row.final_folder = REVIEW
                continue
            row.jev_choice = o.choice or ""
            row.jev_probabilities = o.probabilities
            row.jev_batch_id = o.batch_id or ""
            row.jev_batch_input_tokens = o.batch_input_tokens
            row.final_folder = _folder_for(o.decision, refs_by_key) if o.decision != JEV_REVIEW else REVIEW
            if o.error:
                row.error = o.error


def similarity_matrix(
    files: list[ImageFile], refs: list[Reference], cfg: Config, embedder: Embedder
) -> tuple[np.ndarray, list[ImageFile]]:
    """Scores for calibration: returns (N, M) matrix and the files that embedded successfully."""
    cache = EmbeddingCache(cfg.embed.cache_dir, embedder.tag)
    R = embed_references(embedder, cache, [r.image for r in refs], cfg.embed.flip_refs, cfg.embed.batch_size)
    vectors, _ = embed_candidates(embedder, cache, files, cfg.embed.batch_size)
    ok = [f for f in files if f.sha256 in vectors]
    if not ok:
        return np.zeros((0, len(refs)), dtype=np.float32), ok
    return cosine_matrix(np.stack([vectors[f.sha256] for f in ok]), R), ok
