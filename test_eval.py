"""Tests for the parts of the harness that fail silently rather than loudly."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
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


def _run_main(monkeypatch, raw_dir, out_dir, extra_args=None) -> int:
    argv = ["build_splits.py", "--split-date", "2026-03-01"]
    argv.extend(extra_args or [])
    monkeypatch.setattr(build_splits, "RAW", raw_dir)
    monkeypatch.setattr(build_splits, "OUT", out_dir)
    monkeypatch.setattr(sys, "argv", argv)
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


def _read_split_frames(out_dir):
    train = pd.read_csv(out_dir / "train.csv")
    test = pd.read_csv(out_dir / "test.csv")
    for frame in (train, test):
        frame["first_seen"] = pd.to_datetime(
            frame["first_seen"], utc=True, format="mixed"
        )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return train, test, manifest


def _shape_neutral_rows(n_benign, n_phish, benign_stamp, phish_stamps):
    """Rows sharing one URL-shape pool so the leakage audit stays 'ok'.

    Benign domains are ``good<i>-probe.com`` and phish ``evil<i>-probe.com``;
    templates round-robin so shape alone cannot separate the classes.
    """
    templates = ["/", "/about", "/a/b/c?x=1&y=2", "/news/2024/06/15/story"]
    rows = []
    for i in range(n_benign):
        rows.append(
            {
                "url": f"https://good{i:03d}-probe.com{templates[i % 4]}",
                "label": 0,
                "first_seen": benign_stamp,
                "source": "probe",
            }
        )
    for i in range(n_phish):
        rows.append(
            {
                "url": f"https://evil{i:03d}-probe.com{templates[i % 4]}",
                "label": 1,
                "first_seen": phish_stamps[i],
                "source": "probe",
            }
        )
    return rows


def test_phish_positives_split_temporally(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    stamps = ["2026-01-05T00:00:00+00:00"] * 60 + ["2026-06-05T00:00:00+00:00"] * 60
    rows = _shape_neutral_rows(120, 120, "2026-06-05T00:00:00+00:00", stamps)
    _write_raw_log(raw_dir, rows)

    assert _run_main(monkeypatch, raw_dir, out_dir) == 0
    train, test, _ = _read_split_frames(out_dir)
    cutoff = pd.Timestamp("2026-03-01", tz="UTC")

    train_phish = train[train.label == 1]
    test_phish = test[test.label == 1]
    assert len(train_phish) > 0 and len(test_phish) > 0
    assert (train_phish["first_seen"] < cutoff).all()
    assert (test_phish["first_seen"] >= cutoff).all()


def test_benign_split_ignores_crawl_timestamp(tmp_path, monkeypatch):
    # The reported bug: benign rows all carry the crawl date, so a global
    # time cutoff strands every negative in test and starves train. Benign
    # negatives must appear on both sides regardless of their timestamps.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    stamps = ["2026-01-05T00:00:00+00:00"] * 60 + ["2026-06-05T00:00:00+00:00"] * 60
    rows = _shape_neutral_rows(120, 120, "2026-06-05T00:00:00+00:00", stamps)
    _write_raw_log(raw_dir, rows)

    assert _run_main(monkeypatch, raw_dir, out_dir) == 0
    train, test, _ = _read_split_frames(out_dir)

    for frame in (train, test):
        assert (frame.label == 0).sum() > 0
        assert (frame.label == 1).sum() > 0


def test_benign_assignment_deterministic_across_timestamps(tmp_path, monkeypatch):
    # Same benign domains/URLs, different crawl stamps: the domain-hash
    # partition must be identical, proving timestamps are not consulted.
    stamps = ["2026-01-05T00:00:00+00:00"] * 60 + ["2026-06-05T00:00:00+00:00"] * 60
    early = _shape_neutral_rows(120, 120, "2026-01-05T00:00:00+00:00", stamps)
    late = _shape_neutral_rows(120, 120, "2026-06-05T00:00:00+00:00", stamps)
    raw_early, out_early = tmp_path / "raw_early", tmp_path / "out_early"
    raw_late, out_late = tmp_path / "raw_late", tmp_path / "out_late"
    _write_raw_log(raw_early, early)
    _write_raw_log(raw_late, late)

    assert _run_main(monkeypatch, raw_early, out_early) == 0
    assert _run_main(monkeypatch, raw_late, out_late) == 0
    train_e, test_e, _ = _read_split_frames(out_early)
    train_l, test_l, _ = _read_split_frames(out_late)

    for col in ("train", "test"):
        frame_e = train_e if col == "train" else test_e
        frame_l = train_l if col == "train" else test_l
        dom_e = set(frame_e[frame_e.label == 0]["registrable_domain"])
        dom_l = set(frame_l[frame_l.label == 0]["registrable_domain"])
        assert dom_e == dom_l


def test_repeated_runs_produce_identical_splits(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
    _write_raw_log(raw_dir, _clean_rows())

    assert _run_main(monkeypatch, raw_dir, out_a) == 0
    assert _run_main(monkeypatch, raw_dir, out_b) == 0
    train_a, test_a, _ = _read_split_frames(out_a)
    train_b, test_b, _ = _read_split_frames(out_b)

    pd.testing.assert_frame_equal(
        train_a.sort_values("url").reset_index(drop=True),
        train_b.sort_values("url").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        test_a.sort_values("url").reset_index(drop=True),
        test_b.sort_values("url").reset_index(drop=True),
    )


def test_no_domain_overlap_and_whole_domains_grouped(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    rows = _clean_rows()
    # One benign domain contributing several URLs must stay on a single side.
    for path in ("/a", "/b", "/c"):
        rows.append(
            {
                "url": f"https://multilink-probe.com{path}",
                "label": 0,
                "first_seen": "2026-06-05T00:00:00+00:00",
                "source": "probe",
            }
        )
    _write_raw_log(raw_dir, rows)

    assert _run_main(monkeypatch, raw_dir, out_dir) == 0
    train, test, _ = _read_split_frames(out_dir)

    assert not (set(train["registrable_domain"]) & set(test["registrable_domain"]))
    for frame in (train, test):
        assert set(frame.label.unique()) == {0, 1}
    multi = pd.concat([train, test])
    multi = multi[multi["registrable_domain"] == "multilink-probe.com"]
    assert len(multi) == 3
    assert (multi["registrable_domain"] == "multilink-probe.com").all()
    assert (train["registrable_domain"] == "multilink-probe.com").sum() in (0, 3)


def test_neg_domain_hash_rule_is_stable_and_documented():
    import hashlib

    domain = "example-probe.com"
    seed, fraction = "phishnet-neg-split-v1", 0.2
    expected = (
        int.from_bytes(hashlib.sha256(f"{seed}:{domain}".encode()).digest()[:8], "big")
        / 2**64
    )
    assert build_splits.neg_domain_hash_fraction(domain, seed) == expected
    assert build_splits.neg_domain_is_test(domain, seed, fraction) == (
        expected < fraction
    )
    # Stable: repeated calls agree (hash() would not guarantee this).
    assert build_splits.neg_domain_is_test(
        domain, seed, fraction
    ) == build_splits.neg_domain_is_test(domain, seed, fraction)


def test_manifest_records_benign_split_contract(tmp_path, monkeypatch):
    # Reproducibility contract: the manifest must pin the benign partition
    # config and reconcile per-class counts with the written splits.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    _write_raw_log(raw_dir, _clean_rows())

    assert _run_main(monkeypatch, raw_dir, out_dir) == 0
    train, test, manifest = _read_split_frames(out_dir)

    benign_split = manifest["benign_split"]
    assert benign_split["method"] == "registrable-domain-hash"
    assert benign_split["seed"] == build_splits.NEG_HASH_SEED_DEFAULT
    assert benign_split["test_fraction"] == build_splits.NEG_TEST_FRACTION_DEFAULT
    assert manifest["phish_temporal_cutoff"] == manifest["split_date"]
    assert (
        manifest["n_train"]
        == len(train)
        == (manifest["n_train_phish"] + manifest["n_train_benign"])
    )
    assert (
        manifest["n_test"]
        == len(test)
        == (manifest["n_test_phish"] + manifest["n_test_benign"])
    )
    assert manifest["n_train_benign"] > 0 and manifest["n_test_benign"] > 0
    assert manifest["n_train_phish"] > 0 and manifest["n_test_phish"] > 0
