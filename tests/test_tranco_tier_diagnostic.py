"""Tests for the Tranco rank-tier hostname diagnostic.

The diagnostic lives in ``scratch/tranco_tier_diagnostic.py`` (a
non-packaged script, hence loaded by file path). These tests pin:

* the frozen Tranco input's sha256 (offline read);
* reproducible, in-bounds, non-overlapping tier sampling;
* metric delegation to the existing project implementations;
* full-pipeline execution with sockets blocked (no network possible);
* presence of the required scope limitation in the report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import socket as _socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_splits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_diagnostic() -> Any:
    path = REPO_ROOT / "scratch" / "tranco_tier_diagnostic.py"
    spec = importlib.util.spec_from_file_location("tranco_tier_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tranco_tier_diagnostic"] = module
    spec.loader.exec_module(module)
    return module


diag = _load_diagnostic()


class _BlockedSocket(_socket.socket):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("network access blocked in test")


def test_frozen_tranco_input_sha_matches() -> None:
    """The pinned 1M artifact reads offline with the recorded digest."""
    path = REPO_ROOT / "data" / "raw" / "tranco-46VQX-top1000000-2026-09-13.csv"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == diag.TRANC0_SHA256
    assert diag.TRANC0_ID == "46VQX"


def test_tier_boundaries_are_documented_and_disjoint() -> None:
    """Audit-cleanliness: A = 10,000-99,999, B = 100,000-1,000,000."""
    assert diag.TIER_A == (10_000, 99_999)
    assert diag.TIER_B == (100_000, 1_000_000)
    assert diag.TIER_A[1] < diag.TIER_B[0]  # no shared rank
    assert "subdomain_count" not in diag.EVIDENCE_METRICS


def _rank_map(n: int) -> dict[int, str]:
    return {r: f"host{r}.example.com" for r in range(1, n + 1)}


def test_tier_sampling_bounds_and_determinism() -> None:
    mapping = _rank_map(200)
    first = diag.sample_tier(mapping, (10, 100), 20, seed=7)
    second = diag.sample_tier(mapping, (10, 100), 20, seed=7)
    other_seed = diag.sample_tier(mapping, (10, 100), 20, seed=8)
    assert first == second  # fixed seed -> identical sample
    assert len(first) == 20
    assert all(10 <= int(h.split("host")[1].split(".")[0]) <= 100 for h in first)
    tier_b = diag.sample_tier(mapping, (101, 200), 20, seed=7)
    assert not (set(first) & set(tier_b))  # tiers never overlap
    assert first == sorted(first)  # stable output order
    assert other_seed != []  # other seed sampled without error


def test_hostname_metrics_use_existing_implementations() -> None:
    urls = [
        "https://www.example.com/path?q=1",
        "https://example-12.com/",
        "https://deep.sub.example.com/x/y",
    ]
    got = diag.hostname_metrics(urls)
    expected_len = build_splits.shape_features(pd.DataFrame({"url": urls}))[
        :, diag.NETLOC_LEN_INDEX
    ]
    assert np.array_equal(got["netloc_len"], expected_len)

    # Subdomain formula parity with comprehensive_phishing_features.
    assert list(got["subdomain_count"]) == [1.0, 0.0, 2.0]

    # Densities are ratios over the same urlparse netloc string.
    assert got["hyphen_density"][1] == (
        "example-12.com".count("-") / len("example-12.com")
    )
    assert got["digit_density"][1] == (
        sum(c.isdigit() for c in "example-12.com") / len("example-12.com")
    )
    assert got["hyphen_density"][0] == 0.0


def test_describe_reports_required_keys() -> None:
    stats = diag.describe(np.array([4.0, 8.0, 15.0, 16.0, 23.0, 42.0]))
    for key in ("n", "mean", "median", "p25", "p75", "p90", "p95"):
        assert key in stats, key
    assert stats["n"] == 6
    assert stats["median"] == float(np.median([4, 8, 15, 16, 23, 42]))


def _write_mini_inputs(base: Path) -> tuple[Path, Path]:
    tranco = base / "mini-top.csv"
    tranco.write_text(
        "".join(f"{r},host{r}.example.com\n" for r in range(1, 61)),
        encoding="utf-8",
    )
    rows = [
        ("https://phish-a.com/login", 1, "phish-a.com"),
        ("https://phish-b.com/", 1, "phish-b.com"),
        ("https://b1.com/", 0, "b1.com"),
        ("https://b2.com/about", 0, "b2.com"),
    ]
    eval_test = base / "test.csv"
    pd.DataFrame(rows, columns=["url", "label", "registrable_domain"]).to_csv(
        eval_test, index=False
    )
    return tranco, eval_test


def test_pipeline_runs_offline_with_scope(tmp_path: Path, monkeypatch: Any) -> None:
    """Full pipeline on mini inputs with every socket blocked."""
    monkeypatch.setattr(_socket, "socket", _BlockedSocket)
    monkeypatch.setattr(diag, "TIER_A", (10, 20))
    monkeypatch.setattr(diag, "TIER_B", (21, 60))
    tranco, eval_test = _write_mini_inputs(tmp_path)
    h = hashlib.sha256(tranco.read_bytes()).hexdigest()
    monkeypatch.setattr(diag, "TRANC0_SHA256", h)

    out = tmp_path / "report.json"
    first = diag.run_diagnostic(
        tranco_path=tranco,
        eval_test=eval_test,
        n_per_tier=5,
        seed=0,
        output_json=out,
    )
    second = diag.run_diagnostic(
        tranco_path=tranco,
        eval_test=eval_test,
        n_per_tier=5,
        seed=0,
        output_json=None,
    )
    assert first["populations"] == second["populations"]  # deterministic
    assert "hostname/netloc-shape characteristics only" in (first["scope_limitation"])
    assert "reference population" in first["reference_note"]
    assert first["reproducibility"]["sample_sizes"] == {
        "tier_a": 5,
        "tier_b": 5,
        "phishing": 2,
        "current_benign": 2,
    }
    assert out.is_file()
    for metric in (
        "netloc_len",
        "subdomain_count",
        "hyphen_density",
        "digit_density",
    ):
        assert metric in first["populations"]["tranco_10k_99k"]
        assert metric in first["comparison_vs_phishing"]
