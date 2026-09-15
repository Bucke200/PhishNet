"""Tests for the scheme-composition hard gate in validate_cc_benign.

The contingency counts reproduce the partial-cache trial exactly
(trial2 report: benign 2483/2557 https, phishing 69043/75833 https —
97.11% vs 91.05%, a 0.0606 rate gap). URL lists are generated in memory
from those counts: no network, no cache reads, no fixture files.

Gating convention under test: binary features gate on the rate gap
(|benign_rate - phishing_rate| <= SCHEME_RATE_GAP_MAX), continuous
features gate on ROC-AUC. The trial's 0.0606 gap exceeds the 0.04
tolerance, so the gate fires on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import validate_cc_benign as V

# Exact partial-cache trial contingency (see module docstring).
TRIAL_BENIGN_HTTPS = 2483
TRIAL_BENIGN_N = 2557
TRIAL_PHISH_HTTPS = 69043
TRIAL_PHISH_N = 75833
TRIAL_GAP = abs(TRIAL_BENIGN_HTTPS / TRIAL_BENIGN_N - TRIAL_PHISH_HTTPS / TRIAL_PHISH_N)


def _urls(n_https: int, n_http: int, host: str, path: str = "/x") -> list[str]:
    out = [f"https://{host}{i}.example.com{path}" for i in range(n_https)]
    out += [f"http://{host}{i}.example.net{path}" for i in range(n_http)]
    return out


def _write_urls(path: Path, urls: list[str]) -> None:
    rows = [{"url": u, "label": 1} for u in urls]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_scheme_gate_fires_on_trial_split() -> None:
    """The trial's 0.0606 gap exceeds the 0.04 tolerance: breach."""
    benign = _urls(TRIAL_BENIGN_HTTPS, TRIAL_BENIGN_N - TRIAL_BENIGN_HTTPS, "b")
    phish = _urls(TRIAL_PHISH_HTTPS, TRIAL_PHISH_N - TRIAL_PHISH_HTTPS, "p")
    gap, b_rate, p_rate = V.binary_rate_gap(benign, phish, V.is_https)
    assert gap == pytest.approx(TRIAL_GAP, abs=1e-9)
    assert gap == pytest.approx(0.0606, abs=1e-3)
    assert b_rate == pytest.approx(TRIAL_BENIGN_HTTPS / TRIAL_BENIGN_N)
    assert p_rate == pytest.approx(TRIAL_PHISH_HTTPS / TRIAL_PHISH_N)
    assert gap > V.SCHEME_RATE_GAP_MAX
    # The AUC view agrees (0.5 + gap / 2) and is reported, not gated.
    auc, _, _ = V.scheme_only_auc(benign, phish)
    assert auc == pytest.approx(0.5 + gap / 2, abs=1e-3)


def test_scheme_gate_fires_on_wide_split() -> None:
    """100%-https benign vs 50/50 phishing: gap 0.5, breach."""
    benign = _urls(500, 0, "b")
    phish = _urls(250, 250, "p")
    gap, _, _ = V.binary_rate_gap(benign, phish, V.is_https)
    assert gap == pytest.approx(0.5)
    assert gap > V.SCHEME_RATE_GAP_MAX


def test_scheme_gate_fires_on_all_https_benign() -> None:
    """All-https benign vs 91%-https phishing: gap ~0.09, breach.

    This is the CC-bias direction an AUC-style 0.55 gate could never
    catch (it scores 0.545 there); the gap gate does.
    """
    benign = _urls(500, 0, "b")
    phish = _urls(TRIAL_PHISH_HTTPS, TRIAL_PHISH_N - TRIAL_PHISH_HTTPS, "p")
    gap, _, _ = V.binary_rate_gap(benign, phish, V.is_https)
    assert gap == pytest.approx(1.0 - TRIAL_PHISH_HTTPS / TRIAL_PHISH_N, abs=1e-9)
    assert gap > V.SCHEME_RATE_GAP_MAX


def test_scheme_gate_passes_on_matched_rates() -> None:
    """Identical scheme rates: gap 0, pass."""
    benign = _urls(910, 90, "b")
    phish = _urls(910, 90, "p")
    gap, _, _ = V.binary_rate_gap(benign, phish, V.is_https)
    assert gap == pytest.approx(0.0)
    assert gap <= V.SCHEME_RATE_GAP_MAX


def test_scheme_gate_fails_validation_end_to_end(tmp_path: Path) -> None:
    """main() exits 1 with gap + rates in the report (pass or fail).

    Hermetic: the phishing reference is a synthetic tmp file, never the
    live data/raw glob. The validator's default --phish-glob reads every
    matching snapshot, so any pinned rate rots with each daily feed —
    that is exactly how this test broke when the 2026-09-15 snapshots
    landed (reference moved 0.9105 -> 0.9090).
    """
    rows: list[dict[str, Any]] = [
        {"url": f"http://cand{i}.example.com/page", "label": 0} for i in range(200)
    ]
    benign = tmp_path / "cand.jsonl"
    benign.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    phish_urls = [f"https://ph{i}.example.net/x" for i in range(182)]
    phish_urls += [f"http://ph{i}.example.org/x" for i in range(18)]
    phish = tmp_path / "phish.jsonl"
    _write_urls(phish, phish_urls)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(benign),
            "--phish-glob",
            str(phish),
            "--out",
            str(report),
        ]
    )
    assert rc == 1
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert any(f.startswith("scheme_rate_gap=") for f in rep["failures"])
    scheme = rep["scheme_rates"]
    assert scheme["benign_is_https_overall"] == pytest.approx(0.0)
    assert scheme["phish_is_https_overall"] == pytest.approx(0.91)
    assert scheme["scheme_rate_gap"] == pytest.approx(0.91)
    assert scheme["scheme_rate_gap_max"] == V.SCHEME_RATE_GAP_MAX
    assert scheme["scheme_gate_passed"] is False
    assert "has_port_gap" in scheme
