"""Configuration: defaults, TOML loading, CLI overrides.

All thresholds here are placeholders until calibration (IMPLEMENTATION_PLAN.md §6).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Jev image limits (plan §2). Kept here so preprocess and the judge agree.
JEV_MIN_SIDE = 280
JEV_MAX_SIDE = 1568
JEV_MIN_PIXELS = 78_400
JEV_MAX_PIXELS = 2_458_624
JEV_MAX_IMAGE_BYTES = 16 * 1024 * 1024
JEV_MAX_REQUEST_BYTES = 32 * 1024 * 1024


@dataclass
class ScanConfig:
    recursive: bool = True


@dataclass
class EmbedConfig:
    model: str = "dinov2-base"  # dinov2-base | siglip-base
    patch_mean: bool = False  # concatenate mean-pooled patch tokens to CLS
    flip_refs: bool = True  # average reference with its horizontal flip
    batch_size: int = 16
    device: str = "auto"  # auto | cpu | cuda | mps
    cache_dir: str = ".cache/embeddings"


@dataclass
class BandConfig:
    t_high: float = 0.70
    t_low: float = 0.45
    margin: float = 0.05


@dataclass
class JevConfig:
    enabled: bool = True
    # Verified live 2026-10. "jev-1.13"/"jev-1.14" are rejected as unknown. "latest" is a
    # moving alias: pin a versioned id once known, or answers can shift under a calibrated accept.
    model: str = "jev-latest"
    # Endpoint and auth scheme are taken from the plan; verify against current TypeSafe docs.
    base_url: str = "https://api.typesafe.ai"
    endpoint: str = "/v1/systemone"
    api_key_env: str = "TYPESAFE_API_KEY"
    accept: float = 0.80
    max_batch: int = 16
    max_request_mib: float = 24.0
    max_side: int = 1024
    jpeg_quality: int = 90
    max_requests: int = 50
    max_retries: int = 3
    timeout_s: float = 120.0
    long_context: bool = False
    usd_per_mtok_input: float = 0.042  # confirm on the pricing page


@dataclass
class Config:
    scan: ScanConfig = field(default_factory=ScanConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    band: BandConfig = field(default_factory=BandConfig)
    jev: JevConfig = field(default_factory=JevConfig)

    def validate(self) -> None:
        b, j = self.band, self.jev
        if not (0.0 <= b.t_low <= b.t_high <= 1.0):
            raise ValueError(f"need 0 <= t_low <= t_high <= 1, got t_low={b.t_low} t_high={b.t_high}")
        if b.margin < 0:
            raise ValueError("margin must be >= 0")
        if not (0.0 <= j.accept <= 1.0):
            raise ValueError("jev.accept must be in [0, 1]")
        if not (JEV_MIN_SIDE <= j.max_side <= JEV_MAX_SIDE):
            raise ValueError(f"jev.max_side must be in [{JEV_MIN_SIDE}, {JEV_MAX_SIDE}]")
        if j.max_batch < 1 or j.max_requests < 0:
            raise ValueError("jev.max_batch must be >= 1 and jev.max_requests >= 0")
        if not (0 < j.max_request_mib * 1024 * 1024 <= JEV_MAX_REQUEST_BYTES):
            raise ValueError("jev.max_request_mib must be in (0, 32]")
        if not (1 <= j.jpeg_quality <= 95):
            raise ValueError("jev.jpeg_quality must be in [1, 95]")
        if self.embed.batch_size < 1:
            raise ValueError("embed.batch_size must be >= 1")


def _merge(obj: Any, data: dict[str, Any], where: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {where}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config section {where}{key} must be a table")
            _merge(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, type(current)(value))


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load defaults, then the TOML file, then dotted overrides like {"band.t_high": 0.8}.

    Overrides whose value is None are ignored, so CLI options can default to None.
    """
    cfg = Config()
    if path is not None:
        with open(path, "rb") as fh:
            _merge(cfg, tomllib.load(fh), "")
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        _merge(cfg, {section: {key: value}}, "")
    cfg.validate()
    return cfg
