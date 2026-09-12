"""Tests for the parts of the harness that fail silently rather than loudly."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import build_splits
from build_splits import normalise
from eval import (
    calibration,
    precision_at_prevalence,
    rates_at,
    threshold_at_fpr,
)


def test_threshold_respects_fpr_budget_with_ties():
    # 100 negatives, 50 of them tied at exactly 0.9. A budget of 5 false
    # positives cannot be spent inside that tie, so the threshold must move
    # above it and the achieved FPR must stay within budget.
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.full(50, 0.9), np.linspace(0, 0.5, 50), np.full(10, 0.95)])
    thr = threshold_at_fpr(y, s, 0.05)
    assert rates_at(y, s, thr)["fpr"] <= 0.05


def test_threshold_when_budget_is_zero():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
    thr = threshold_at_fpr(y, s, 0.005)  # 0.005 * 50 = 0 allowed FPs
    assert rates_at(y, s, thr)["fpr"] == 0.0


def test_threshold_never_exceeds_budget_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 500)
        s = rng.random(500).round(2)  # heavy ties
        for target in (0.001, 0.01, 0.05):
            thr = threshold_at_fpr(y, s, target)
            assert rates_at(y, s, thr)["fpr"] <= target + 1e-12


def test_perfect_separation_gives_full_recall():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.zeros(100), np.ones(100)])
    thr = threshold_at_fpr(y, s, 0.005)
    assert rates_at(y, s, thr)["recall"] == 1.0


def test_precision_at_prevalence_collapses_when_rare():
    # 95% recall, 0.5% FPR looks great on a balanced test set and is unusable
    # at real browsing prevalence. This is the number that belongs in the README.
    assert precision_at_prevalence(0.95, 0.005, 0.5) > 0.99
    assert precision_at_prevalence(0.95, 0.005, 1e-4) < 0.02


def test_calibration_of_calibrated_scores_is_near_zero():
    rng = np.random.default_rng(1)
    s = rng.random(20_000)
    y = (rng.random(20_000) < s).astype(int)
    cal = calibration(y, s, bins=10)
    assert cal["ece"] < 0.02


def test_calibration_of_overconfident_scores_is_large():
    rng = np.random.default_rng(2)
    s = np.clip(rng.random(5_000) * 0.4 + 0.6, 0, 1)
    y = (rng.random(5_000) < 0.1).astype(int)
    assert calibration(y, s, bins=10)["ece"] > 0.4


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://Example.COM:80/Path?a=1#frag", "http://example.com/Path?a=1"),
        ("https://Example.com:443/", "https://example.com/"),
        ("https://example.com", "https://example.com/"),
        # userinfo must survive normalisation — stripping it here would hide the
        # very spoofing pattern the detector needs to see.
        ("https://paypal.com@evil.tk/login", "https://paypal.com@evil.tk/login"),
        ("ftp://example.com/x", None),
        ("not a url", None),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def _write_raw_log(raw_dir, rows: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "probe-2026-06-05.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _run_main(monkeypatch, raw_dir, out_dir) -> int:
    monkeypatch.setattr(build_splits, "RAW", raw_dir)
    monkeypatch.setattr(build_splits, "OUT", out_dir)
    monkeypatch.setattr(sys, "argv", ["build_splits.py", "--split-date", "2026-03-01"])
    return build_splits.main()


def _leaking_rows() -> list[dict]:
    # Bare benign domains vs long deep phishing URLs: URL shape alone
    # separates the classes, so the audit must report LEAKING.
    rows = []
    for i in list(range(60)) + list(range(60, 100)):
        day = "2026-01-05" if i < 60 else "2026-06-05"
        stamp = f"{day}T00:00:00+00:00"
        rows.append(
            {
                "url": f"https://benign{i:03d}-probe.com/",
                "label": 0,
                "first_seen": stamp,
                "source": "probe",
            }
        )
        rows.append(
            {
                "url": f"https://phish{i:03d}-probe.com/login/verify/account/update/index.php?session={i}&token=abc{i}",  # noqa: E501
                "label": 1,
                "first_seen": stamp,
                "source": "probe",
            }
        )
    return rows


def _clean_rows() -> list[dict]:
    # Both classes share the same URL-shape pool ("good"/"evil" host labels
    # are the same length, templates round-robin), so shape alone cannot
    # separate them and the audit must report ok.
    templates = ["/", "/about", "/a/b/c?x=1&y=2", "/news/2024/06/15/story"]
    rows = []
    for i in list(range(100)) + list(range(100, 150)):
        day = "2026-01-05" if i < 100 else "2026-06-05"
        stamp = f"{day}T00:00:00+00:00"
        t = templates[i % 4]
        rows.append(
            {
                "url": f"https://good{i:03d}-probe.com{t}",
                "label": 0,
                "first_seen": stamp,
                "source": "probe",
            }
        )
        rows.append(
            {
                "url": f"https://evil{i:03d}-probe.com{t}",
                "label": 1,
                "first_seen": stamp,
                "source": "probe",
            }
        )
    return rows


def test_main_leaking_halts_without_writing_splits(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    _write_raw_log(raw_dir, _leaking_rows())

    rc = _run_main(monkeypatch, raw_dir, out_dir)

    assert rc != 0
    assert "LEAKING" in capsys.readouterr().err
    assert not (out_dir / "train.csv").exists()
    assert not (out_dir / "test.csv").exists()
    assert not (out_dir / "manifest.json").exists()


def test_main_ok_split_writes_outputs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    _write_raw_log(raw_dir, _clean_rows())

    rc = _run_main(monkeypatch, raw_dir, out_dir)

    assert rc == 0
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert (out_dir / "manifest.json").exists()
