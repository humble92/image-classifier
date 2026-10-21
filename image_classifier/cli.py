"""typer entry point: ``image-classifier classify``, ``calibrate``, and ``jev``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import calibration
from .config import Config, JevConfig, load_config
from .embed import load_embedder
from .io_scan import load_refs
from .jev_judge import JevClient, JevError, JevItem, JevJudge, JevRef
from .pipeline import classify as run_classify
from .pipeline import similarity_matrix
from .preprocess import JevImageError, jev_jpeg
from .report import summarize
from .route import OutputExistsError

app = typer.Typer(add_completion=False, help="Sort images into per-reference folders by identity.")

# Indirection so tests can inject a fake embedder without downloading weights.
embedder_factory = load_embedder


def _default_jev_client(cfg: JevConfig, api_key: str) -> JevClient:
    return JevClient(cfg, api_key)


# Indirection so tests can inject a mock HTTP client without calling TypeSafe.
jev_client_factory = _default_jev_client

RefsOpt = Annotated[
    list[Path], typer.Option("--refs", help="Reference image file or directory (repeatable, 1-5 images total)")
]
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", exists=True, dir_okay=False, help="TOML config")]
ModelOpt = Annotated[Optional[str], typer.Option("--model", help="dinov2-base | siglip-base")]
PatchOpt = Annotated[Optional[bool], typer.Option("--patch-mean/--cls-only", help="Append mean patch tokens")]


def _load(config: Path | None, overrides: dict) -> Config:
    try:
        return load_config(config, overrides)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _embedder(cfg: Config):
    try:
        return embedder_factory(cfg.embed)
    except (OSError, ValueError, ImportError) as exc:
        # Most often: no network on first run, so the weights cannot be downloaded.
        typer.echo(f"error: cannot load embedding model {cfg.embed.model!r}: {str(exc).splitlines()[0]}", err=True)
        typer.echo("hint: the first run downloads weights from huggingface.co; check network or HF_HOME.", err=True)
        raise typer.Exit(2) from exc


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def classify(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Directory of candidate images")],
    refs: RefsOpt,
    out: Annotated[Path, typer.Option("--out", help="Output directory")],
    config: ConfigOpt = None,
    model: ModelOpt = None,
    patch_mean: PatchOpt = None,
    t_high: Annotated[Optional[float], typer.Option("--t-high")] = None,
    t_low: Annotated[Optional[float], typer.Option("--t-low")] = None,
    margin: Annotated[Optional[float], typer.Option("--margin")] = None,
    no_jev: Annotated[bool, typer.Option("--no-jev", help="Route borderline images to _review/")] = False,
    jev_model: Annotated[Optional[str], typer.Option("--jev-model", help="Jev model id (default jev-latest)")] = None,
    jev_accept: Annotated[Optional[float], typer.Option("--jev-accept")] = None,
    jev_batch: Annotated[Optional[int], typer.Option("--jev-batch")] = None,
    max_jev_requests: Annotated[Optional[int], typer.Option("--max-jev-requests")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Bands + estimated Jev requests only")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Clear folders a previous run created")] = False,
    no_recursive: Annotated[bool, typer.Option("--no-recursive")] = False,
) -> None:
    """Sort INPUT_DIR into one folder per reference identity."""
    cfg = _load(
        config,
        {
            "embed.model": model,
            "embed.patch_mean": patch_mean,
            "band.t_high": t_high,
            "band.t_low": t_low,
            "band.margin": margin,
            "jev.enabled": False if no_jev else None,
            "jev.model": jev_model,
            "jev.accept": jev_accept,
            "jev.max_batch": jev_batch,
            "jev.max_requests": max_jev_requests,
            "scan.recursive": False if no_recursive else None,
        },
    )
    embedder = _embedder(cfg)
    try:
        result = run_classify(input_dir, refs, out, cfg, embedder, dry_run=dry_run, overwrite=overwrite)
    except (OutputExistsError, ValueError, FileNotFoundError, NotADirectoryError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    typer.echo("References: " + ", ".join(f"{r.key}={r.name}" for r in result.refs))
    typer.echo(
        summarize(
            result.rows,
            result.jev_requests,
            result.input_tokens,
            cfg.jev.usd_per_mtok_input,
            dry_run=dry_run,
            estimated_requests=result.estimated_requests,
        )
    )
    for note in result.notes:
        typer.echo(f"note: {note}", err=True)
    if not dry_run:
        typer.echo(f"Report: {out / 'report.csv'}")


@app.command()
def calibrate(
    labeled_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="LABELED_DIR/<ref_name|none>/")],
    refs: RefsOpt,
    config: ConfigOpt = None,
    model: ModelOpt = None,
    patch_mean: PatchOpt = None,
    min_precision: Annotated[float, typer.Option("--min-precision")] = 0.98,
    max_recall_loss: Annotated[float, typer.Option("--max-recall-loss")] = 0.02,
    top: Annotated[int, typer.Option("--top")] = 10,
) -> None:
    """Sweep t_high / t_low / margin on a labeled set and print the cheapest configs that meet the targets."""
    cfg = _load(config, {"embed.model": model, "embed.patch_mean": patch_mean})
    try:
        references = load_refs(refs)
        files, labels = calibration.load_labeled(labeled_dir, references)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    embedder = _embedder(cfg)
    scores, ok = similarity_matrix(files, references, cfg, embedder)
    keep = {f.sha256 for f in ok}
    labels = labels[[f.sha256 in keep for f in files]]
    typer.echo(f"Labeled images: {len(ok)} ({int((labels >= 0).sum())} matches, {int((labels < 0).sum())} none)")
    typer.echo(f"Model: {embedder.tag}")

    current = calibration.evaluate(scores, labels, cfg.band)
    typer.echo("\nCurrent config:")
    typer.echo(calibration.format_results([current]))

    results = calibration.sweep(scores, labels)
    chosen = calibration.choose(results, min_precision, max_recall_loss)
    if not chosen:
        typer.echo(
            f"\nNo config reaches accept precision >= {min_precision} with recall loss <= {max_recall_loss}. "
            "Try another model/feature variant or more reference photos."
        )
        raise typer.Exit(1)
    typer.echo(f"\nTop {min(top, len(chosen))} configs (least borderline first):")
    typer.echo(calibration.format_results(chosen, top))
    b = chosen[0]
    typer.echo(f"\n[band]\nt_high = {b.t_high}\nt_low = {b.t_low}\nmargin = {b.margin}")


@app.command()
def jev(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="One candidate image")],
    refs: RefsOpt,
    config: ConfigOpt = None,
    jev_model: Annotated[Optional[str], typer.Option("--jev-model", help="Jev model id (default jev-latest)")] = None,
    jev_accept: Annotated[Optional[float], typer.Option("--jev-accept")] = None,
) -> None:
    """Send one candidate and the references to Jev. Skips embeddings and banding."""
    cfg = _load(config, {"jev.model": jev_model, "jev.accept": jev_accept, "jev.enabled": True})
    try:
        references = load_refs(refs)
        jev_refs = [
            JevRef(r.key, r.name, jev_jpeg(r.image.path, cfg.jev.max_side, cfg.jev.jpeg_quality)) for r in references
        ]
        item = JevItem(image.name, jev_jpeg(image, cfg.jev.max_side, cfg.jev.jpeg_quality))
        client = jev_client_factory(cfg.jev, os.environ.get(cfg.jev.api_key_env, ""))
    except (JevError, JevImageError, ValueError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    judge = JevJudge(client, jev_refs, cfg.jev)
    outcome = judge.run([item])[item.item_id]
    typer.echo("References: " + ", ".join(f"{r.key}={r.name}" for r in references))
    typer.echo(f"Candidate: {image}")
    typer.echo(f"Model: {cfg.jev.model}")
    typer.echo(f"Requests: {judge.requests_made}")
    typer.echo(f"Input tokens: {judge.input_tokens}")
    if outcome.error:
        typer.echo(f"error: {outcome.error}", err=True)
        raise typer.Exit(1)
    probs = outcome.probabilities or {}
    typer.echo(f"Choice: {outcome.choice}")
    typer.echo(f"Decision: {outcome.decision} (accept >= {cfg.jev.accept})")
    typer.echo("Probabilities: " + json.dumps(probs, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    app()
