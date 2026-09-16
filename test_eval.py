"""Tests for the parts of the harness that fail silently rather than loudly."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import build_splits
import validate_cc_benign
from build_splits import normalise
from eval import (
    STRICT_FPR,
    EvalConfig,
    bootstrap_fpr_interval,
    build_slices,
    calibration,
    collect_warnings,
    evaluate,
    fpr_interval_report,
    paired_bootstrap_ci,
    pr_auc,
    precision_at_prevalence,
    rates_at,
    recall_at_fpr,
    threshold_at_fpr,
    to_markdown,
    wilson_interval,
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
    argv = [
        "build_splits.py",
        "--split-date",
        "2026-03-01",
        # Small synthetic fixtures (dozens of domains) sit far below the
        # committed disjointness gates; disable them here — the gates
        # themselves are covered in tests/test_build_splits_gates.py.
        "--max-straddler-drop-share",
        "1.0",
        "--min-benign-test-domains",
        "0",
    ]
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


def test_straddling_etld1_subdomains_isolated_from_test(tmp_path, monkeypatch):
    # Invariant: one eTLD+1 under different subdomains on both sides of the
    # temporal cutoff must not leak into test. The builder drops straddling
    # registrable domains from test (never raw-host comparison), so
    # login.* (train side) and www.* (test side) of one domain isolate.
    from urllib.parse import urlparse as _urlparse

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    stamps = ["2026-01-05T00:00:00+00:00"] * 60 + ["2026-06-05T00:00:00+00:00"] * 60
    rows = _shape_neutral_rows(120, 120, "2026-06-05T00:00:00+00:00", stamps)
    rows.append(
        {
            "url": "https://login.straddle-probe.com/a/b/c?x=1&y=2",
            "label": 1,
            "first_seen": "2026-01-05T00:00:00+00:00",
            "source": "probe",
        }
    )
    rows.append(
        {
            "url": "https://www.straddle-probe.com/news/2024/06/15/story",
            "label": 1,
            "first_seen": "2026-06-05T00:00:00+00:00",
            "source": "probe",
        }
    )
    _write_raw_log(raw_dir, rows)

    assert _run_main(monkeypatch, raw_dir, out_dir) == 0
    train, test, _ = _read_split_frames(out_dir)

    def _etld1(u: str) -> str:
        host = (_urlparse(str(u)).hostname or "").lower().strip(".")
        e = build_splits.EXTRACT(host)
        return f"{e.domain}.{e.suffix}".lower() if e.suffix and e.domain else host

    assert (
        _etld1("https://login.straddle-probe.com/x")
        == _etld1("https://www.straddle-probe.com/y")
        == "straddle-probe.com"
    )
    assert not (set(train["url"].map(_etld1)) & set(test["url"].map(_etld1)))
    assert "straddle-probe.com" not in set(test["url"].map(_etld1))


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
    assert manifest["psl_snapshot_sha256"] == build_splits.PSL_SNAPSHOT_SHA256
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
    assert set(manifest["raw_file_hashes"]) == {"probe-2026-06-05.jsonl"}
    assert manifest["raw_file_hashes"][
        "probe-2026-06-05.jsonl"
    ] == build_splits.sha256_file(raw_dir / "probe-2026-06-05.jsonl")


def test_raw_and_out_flags_pin_input_set_and_output_dir(tmp_path, monkeypatch):
    # Successor populations (e.g. an eval-heavy split) must be buildable
    # without touching the frozen dirs: --raw selects the exact input set
    # (recorded in the manifest) and --out selects the destination.
    raw_dir = tmp_path / "staged-raw"
    out_dir = tmp_path / "splits-eval"
    default_out = tmp_path / "default-splits"
    _write_raw_log(raw_dir, _clean_rows())

    monkeypatch.setattr(build_splits, "RAW", tmp_path / "unused-raw")
    monkeypatch.setattr(build_splits, "OUT", default_out)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_splits.py",
            "--split-date",
            "2026-03-01",
            "--max-straddler-drop-share",
            "1.0",
            "--min-benign-test-domains",
            "0",
            "--raw",
            str(raw_dir),
            "--out",
            str(out_dir),
        ],
    )
    assert build_splits.main() == 0

    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["raw_files"] == ["probe-2026-06-05.jsonl"]
    # The patched-in defaults must be untouched: nothing lands in OUT.
    assert not default_out.exists()

    train, test, _ = _read_split_frames(out_dir)
    assert manifest["n_train"] == len(train)
    assert manifest["n_test"] == len(test)


def test_benign_test_fraction_is_recorded_and_shifts_negatives(tmp_path, monkeypatch):
    # A larger --benign-test-fraction must move benign mass into test while
    # keeping whole domains together and both classes present on each side.
    raw_dir = tmp_path / "raw"
    out_lo, out_hi = tmp_path / "out-lo", tmp_path / "out-hi"
    _write_raw_log(raw_dir, _clean_rows())

    for out_dir, fraction in ((out_lo, 0.2), (out_hi, 0.8)):
        monkeypatch.setattr(build_splits, "RAW", raw_dir)
        monkeypatch.setattr(build_splits, "OUT", out_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_splits.py",
                "--split-date",
                "2026-03-01",
                "--max-straddler-drop-share",
                "1.0",
                "--min-benign-test-domains",
                "0",
                "--benign-test-fraction",
                str(fraction),
            ],
        )
        assert build_splits.main() == 0

    _, _, manifest_lo = _read_split_frames(out_lo)
    _, _, manifest_hi = _read_split_frames(out_hi)
    assert manifest_lo["benign_split"]["test_fraction"] == 0.2
    assert manifest_hi["benign_split"]["test_fraction"] == 0.8
    assert manifest_hi["n_test_benign"] > manifest_lo["n_test_benign"]
    assert manifest_hi["n_train_benign"] < manifest_lo["n_train_benign"]


def _run_with_flags(monkeypatch, extra_args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_splits.py",
            "--split-date",
            "2026-03-01",
            "--max-straddler-drop-share",
            "1.0",
            "--min-benign-test-domains",
            "0",
            *extra_args,
        ],
    )
    return build_splits.main()


def test_deterministic_manifest_splits_timestamp_into_sidecar(tmp_path, monkeypatch):
    # With --deterministic-manifest the volatile run timestamp must leave
    # manifest.json (which stays a pure function of inputs + flags) and land
    # in run-meta.json instead.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    _write_raw_log(raw_dir, _clean_rows())
    monkeypatch.setattr(build_splits, "RAW", raw_dir)
    monkeypatch.setattr(build_splits, "OUT", out_dir)

    assert _run_with_flags(monkeypatch, ["--deterministic-manifest"]) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "generated_at" not in manifest
    sidecar = json.loads((out_dir / "run-meta.json").read_text())
    assert sidecar["generated_at"]
    assert manifest["n_train"] > 0 and manifest["n_test"] > 0


def test_deterministic_manifest_is_byte_stable_across_runs(tmp_path, monkeypatch):
    # Two runs with the flag must produce byte-identical outputs, so whole
    # directories (train/test/manifest) diff cleanly.
    raw_dir = tmp_path / "raw"
    out_a, out_b = tmp_path / "out-a", tmp_path / "out-b"
    _write_raw_log(raw_dir, _clean_rows())
    monkeypatch.setattr(build_splits, "RAW", raw_dir)

    for out_dir in (out_a, out_b):
        monkeypatch.setattr(build_splits, "OUT", out_dir)
        assert _run_with_flags(monkeypatch, ["--deterministic-manifest"]) == 0

    for name in ("train.csv", "test.csv", "manifest.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_default_manifest_keeps_run_timestamp(tmp_path, monkeypatch):
    # Without the flag, behavior is unchanged: generated_at stays inline and
    # no sidecar is written.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    _write_raw_log(raw_dir, _clean_rows())
    monkeypatch.setattr(build_splits, "RAW", raw_dir)
    monkeypatch.setattr(build_splits, "OUT", out_dir)

    assert _run_with_flags(monkeypatch, []) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["generated_at"]
    assert not (out_dir / "run-meta.json").exists()


def _strict_arrays():
    # 2000 negatives on [0, 0.5], 50 positives at 0.9: budget
    # floor(0.001 * 2000) = 2, exactly attainable.
    neg = np.linspace(0, 0.5, 2000)
    pos = np.full(50, 0.9)
    y = np.array([0] * 2000 + [1] * 50)
    s = np.concatenate([neg, pos])
    return y, s


def test_recall_at_fpr_exact_attainment():
    y, s = _strict_arrays()
    got = recall_at_fpr(y, s)
    assert got["target_fpr"] == STRICT_FPR == 0.001
    assert got["achieved_fpr"] == pytest.approx(0.001)
    assert got["exact"] is True
    assert got["recall"] == pytest.approx(1.0)
    assert got["false_positives"] == 2
    assert got["n_negatives"] == 2000


def test_recall_at_fpr_never_exceeds_budget_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 2000)
        s = rng.random(2000).round(1)  # heavy ties
        got = recall_at_fpr(y, s)
        assert got["achieved_fpr"] <= 0.001 + 1e-12
        # Exactness is honest: a tied top score cannot hit the budget.
        assert got["exact"] == (got["false_positives"] * 1000 == 2000)


def test_recall_at_fpr_small_sample_reports_zero_without_exactness():
    # Fewer than 1000 negatives: zero budget, threshold above the top
    # negative score — a real operating point at empirical FPR 0.
    y = np.array([0] * 500 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.5, 500), np.full(50, 0.9)])
    got = recall_at_fpr(y, s)
    assert got["false_positives"] == 0
    assert got["achieved_fpr"] == 0.0
    assert got["exact"] is False
    assert got["recall"] == pytest.approx(1.0)


def test_recall_at_fpr_threshold_is_empirical_never_interpolated():
    # The reported point must be producible by real scores: the threshold
    # is either an observed score or just above the maximum (budget 0).
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 1500)
        s = rng.random(1500).round(2)
        got = recall_at_fpr(y, s)
        assert got["threshold"] in set(s) or got["threshold"] > s.max()
        assert rates_at(y, s, got["threshold"])["fpr"] <= 0.001 + 1e-12


def test_recall_at_fpr_rejects_nonpositive_budget():
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        recall_at_fpr(y, s, 0.0)


def test_paired_bootstrap_ci_measures_lift_not_overlap():
    # Identical scores: the difference interval sits on zero. A clearly
    # better scorer: the whole interval clears zero — even where the two
    # separate CIs would overlap, the paired interval resolves the lift.
    rng = np.random.default_rng(7)
    y = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"d{i // 4}" for i in range(120)])
    base = rng.random(120)
    lo, hi = paired_bootstrap_ci(pr_auc, y, base, base, 200, 0, groups)
    assert lo <= 0.0 <= hi
    better = base.copy()
    better[y == 1] += 0.3
    lo, hi = paired_bootstrap_ci(pr_auc, y, better, base, 200, 0, groups)
    assert lo > 0.0
    nan_lo, nan_hi = paired_bootstrap_ci(pr_auc, y, base, base, 0, 0, groups)
    assert nan_lo != nan_lo and nan_hi != nan_hi  # nan pair on n_boot<=0


def test_bootstrap_fpr_interval_fixed_threshold():
    # Fixed threshold: resamples measure what the deployed point attains.
    # All-negative scores above thr push the interval to 1; degenerate
    # replicates never crash the helper.
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.linspace(0, 1, 100), np.linspace(0, 1, 100)])
    groups = np.array([f"d{i // 5}" for i in range(200)])
    lo, hi = bootstrap_fpr_interval(y, s, 0.5, 200, 0, groups)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= 0.5 <= hi  # ~half the negatives sit above 0.5
    lo, hi = bootstrap_fpr_interval(y, s, 2.0, 50, 0, groups)
    assert (lo, hi) == (0.0, 0.0)


def test_fpr_verdict_three_valued_on_wider():
    # 21/4321 at 0.5%: Wilson alone sits under budget, but the verdict
    # uses the wider interval — a straddling bootstrap reads
    # indistinguishable, not met.
    y = np.array([0] * 2000 + [1] * 2000)
    s = np.concatenate([np.linspace(0, 1, 2000), np.linspace(0, 1, 2000)])
    groups = np.array([f"d{i // 4}" for i in range(4000)])
    # 0/2000 at 0.5%: the wider interval (Wilson ≈ (0, 0.0019]) fits under
    # budget — met. (At 0/200 Wilson alone already straddles 0.005, which
    # is exactly why small populations read indistinguishable.)
    rep = fpr_interval_report(0, 2000, y, s, 2.0, 200, 0, groups, 0.005)
    assert rep["verdict"] == "met"
    assert rep["wider"][1] <= 0.005
    rep = fpr_interval_report(2000, 2000, y, s, -1.0, 200, 0, groups, 0.005)
    assert rep["verdict"] == "unmet"
    rep = fpr_interval_report(
        1, 200, y, s, float(s[y == 0].max()), 500, 1, groups, 0.005
    )
    assert rep["verdict"] in ("met", "indistinguishable", "unmet")
    assert rep["wider"][0] <= min(rep["wilson"][0], rep["bootstrap"][0])
    assert rep["wider"][1] >= max(rep["wilson"][1], rep["bootstrap"][1])


def test_wilson_interval_covers_rate_and_handles_edges():
    # 21/4321 (the splits-eval budget scale): interval covers the point
    # estimate, stays in [0, 1], and widens as n shrinks.
    lo, hi = wilson_interval(21, 4321)
    assert lo <= 21 / 4321 <= hi
    assert 0.0 <= lo and hi <= 1.0
    narrow = hi - lo
    lo2, hi2 = wilson_interval(2, 462)  # frozen Phase-1 scale
    assert (hi2 - lo2) > narrow
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] > 0.999  # clamped at 1.0 up to fp dust
    assert all(v != v for v in wilson_interval(0, 0))  # nan, nan


def test_survival_stratum_slice_present_when_column_exists():
    # Pre-registered headline rule (docs/point-in-time.md): the unknown
    # stratum stays IN the headline and is disclosed per-stratum — never
    # silently dropped or silently kept. The slice must ride every report
    # whose split carries the column, and stay absent otherwise.
    frame = pd.DataFrame(
        {
            "url": ["https://a.example/", "https://b.example/"],
            "suffix": ["com", "com"],
            "survival_stratum": ["fresh", "unknown"],
        }
    )
    slices = build_slices(frame)
    assert set(slices["survival_stratum"]) == {"fresh", "unknown"}
    assert "survival_stratum" not in build_slices(
        frame.drop(columns=["survival_stratum"])
    )


def test_hosted_slice_reports_tenants_separately():
    # Hosted rows report as their own slice — including across the CSV
    # bool-to-string round-trip, where no row may silently land in
    # "unknown".
    frame = pd.DataFrame(
        {
            "url": ["https://a.example/", "https://b.example/"],
            "suffix": ["com", "com"],
            "is_hosted_tenant": [True, False],
        }
    )
    assert set(build_slices(frame)["hosted"]) == {"hosted-tenant", "other"}
    as_strings = frame.astype({"is_hosted_tenant": str})
    assert set(build_slices(as_strings)["hosted"]) == {"hosted-tenant", "other"}
    assert "hosted" not in build_slices(frame.drop(columns=["is_hosted_tenant"]))


def test_no_builder_shape_halt_below_leaking(tmp_path, monkeypatch, capsys):
    # The 0.60 single-threshold builder gate was proposed and WITHDRAWN
    # (see docs/cc-benign-acquisition.md: replaced by validator-side
    # mechanism hard gates + a 0.70 advisory band). A split clearing the
    # old bands at 0.61 is usable: it writes successfully and refuses
    # nothing. Only a LEAKING verdict halts the builder.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    _write_raw_log(raw_dir, _clean_rows())
    monkeypatch.setattr(
        build_splits,
        "leakage_audit",
        lambda train, test: {"shape_only_roc_auc": 0.61, "verdict": "ok"},
    )
    rc = _run_main(monkeypatch, raw_dir, out_dir)

    assert rc == 0
    assert "SHAPE GATE" not in capsys.readouterr().err
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert (out_dir / "manifest.json").exists()


def test_manifest_has_no_shape_gate_key(tmp_path, monkeypatch):
    # Acceptance outcomes are not manifest keys: the manifest keeps its
    # frozen schema and the gates live in validate_cc_benign.
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "splits"
    _write_raw_log(raw_dir, _clean_rows())
    monkeypatch.setattr(
        build_splits,
        "leakage_audit",
        lambda train, test: {"shape_only_roc_auc": 0.60, "verdict": "ok"},
    )
    assert _run_main(monkeypatch, raw_dir, out_dir) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "shape_gate" not in manifest
    assert "leakage_audit" in manifest


def test_acceptance_gates_live_in_validator():
    # The two-number acceptance spec, pinned in code: mechanism hard
    # gates plus a warn-only shape band (docs/cc-benign-acquisition.md).
    assert validate_cc_benign.SCHEME_RATE_GAP_MAX == 0.04
    assert validate_cc_benign.PATH_DEPTH_AUC_MAXDIST == 0.05
    assert validate_cc_benign.URL_LEN_INVERSION_MIN == 0.0
    assert validate_cc_benign.SHAPE_AUC_ADVISORY == 0.70


class _ProbeScorer:
    name = "probe"

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, urls):  # type: ignore[no-untyped-def]
        return self._scores[: len(urls)]


def _warnings_frame(n_neg: int, n_pos: int) -> pd.DataFrame:
    n = n_neg + n_pos
    return pd.DataFrame(
        {
            "url": [f"https://warn{i:04d}.com/x" for i in range(n)],
            "label": [0] * n_neg + [1] * n_pos,
            "first_seen": ["2026-09-01T00:00:00+00:00"] * n,
            "registrable_domain": [f"warn{i:04d}.com" for i in range(n)],
        }
    )


def test_top_tie_pile_warns_degeneracy() -> None:
    # 25 URLs pile at the top score against a 1-FP budget: the walk cannot
    # spend inside the tie, so the operating point collapses above it.
    # The distinct-scores check sees hundreds of levels and stays silent —
    # this is the second distinct cause of a degenerate row.
    y = np.array([0] * 300 + [1] * 100)
    s = np.concatenate([np.full(25, 1.0), np.linspace(0, 0.5, 275), np.full(100, 0.9)])
    warnings = collect_warnings(_warnings_frame(300, 100), y, s, EvalConfig())
    assert any("collapses above it" in w for w in warnings)


def test_unique_top_score_is_silent() -> None:
    # Same budget, lone maximum: reachable, no pile, no warning.
    y = np.array([0] * 300 + [1] * 100)
    s = np.concatenate([np.linspace(0, 0.999, 300), np.full(100, 0.5)])
    warnings = collect_warnings(_warnings_frame(300, 100), y, s, EvalConfig())
    assert not any("collapses above it" in w for w in warnings)


def test_evaluate_reports_strict_fpr_point(tmp_path):
    # Headline gains the 0.1% operating point; markdown renders it; the
    # configured 0.5% point is unchanged.
    urls = [f"https://example{i:04d}.com/x" for i in range(60)]
    labels = [0] * 40 + [1] * 20
    frame = pd.DataFrame(
        {
            "url": urls,
            "label": labels,
            "first_seen": ["2026-09-01T00:00:00+00:00"] * 60,
            "registrable_domain": [f"example{i:04d}.com" for i in range(60)],
        }
    )
    dataset = tmp_path / "probe.csv"
    frame.to_csv(dataset, index=False)
    scores = [float(i) / 60 for i in range(60)]  # positives score highest
    rep = evaluate(_ProbeScorer(scores), dataset, EvalConfig(bootstrap=0, seed=0))

    h = rep["headline"]
    for key in (
        "roc_auc",
        "recall_at_target_fpr",
        "recall_at_fpr_0_1pct",
        "achieved_fpr_0_1pct",
        "threshold_fpr_0_1pct",
        "fpr_0_1pct_exact",
        "recall_at_fpr_1pct",
        "achieved_fpr_1pct",
        "threshold_fpr_1pct",
        "fpr_1pct_exact",
        "achieved_fpr_wilson",
        "achieved_fpr_0_1pct_wilson",
        "achieved_fpr_1pct_wilson",
    ):
        assert key in h, key
    assert h["achieved_fpr_0_1pct"] <= 0.001 + 1e-12
    assert h["achieved_fpr_1pct"] <= 0.01 + 1e-12
    assert isinstance(h["fpr_0_1pct_exact"], bool)
    assert isinstance(h["fpr_1pct_exact"], bool)
    lo, hi = h["achieved_fpr_wilson"]
    assert lo <= h["achieved_fpr"] <= hi
    md = to_markdown(rep)
    assert "Recall @ FPR≤0.10%" in md
    assert "Recall @ FPR≤1.00%" in md
    assert "Strict point" in md


def _probe_report(tmp_path, stem: str, n_neg: int, n_pos: int, mode=None):  # type: ignore[no-untyped-def]
    urls = [f"https://{stem}{i:04d}.com/x" for i in range(n_neg + n_pos)]
    labels = [0] * n_neg + [1] * n_pos
    frame = pd.DataFrame(
        {
            "url": urls,
            "label": labels,
            "first_seen": ["2026-09-01T00:00:00+00:00"] * (n_neg + n_pos),
            "registrable_domain": [f"{stem}{i:04d}.com" for i in range(n_neg + n_pos)],
        }
    )
    dataset = tmp_path / f"{stem}.csv"
    frame.to_csv(dataset, index=False)
    scores = [float(i) / (n_neg + n_pos) for i in range(n_neg + n_pos)]
    scorer = _ProbeScorer(scores)
    if mode is not None:
        scorer.mode = mode
    return evaluate(scorer, dataset, EvalConfig(bootstrap=0, seed=0))


def _headline_table_lines(md: str) -> list[str]:
    lines = md.splitlines()
    start = lines.index("| Metric | Value | 95% CI (domain bootstrap) | vs baseline |")
    out = []
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        if ln.startswith("| "):
            out.append(ln)
    return out


def test_cross_dataset_baseline_suppresses_deltas(tmp_path):
    # PR-AUC's no-skill floor is the base rate, so a delta between two
    # populations reports lift-over-floor as regression. Different bytes ->
    # different sha256 -> deltas suppressed with a Read-this-first reason.
    rep_a = _probe_report(tmp_path, "alpha", 40, 20)
    rep_b = _probe_report(tmp_path, "beta", 30, 30)
    assert rep_a["dataset"]["sha256"] != rep_b["dataset"]["sha256"]

    md = to_markdown(rep_a, rep_b)
    assert "different populations" in md
    assert "suppressed" in md
    table = _headline_table_lines(md)
    assert len(table) == 10
    assert all(ln.endswith("| — |") for ln in table), table


def test_same_dataset_baseline_keeps_deltas(tmp_path):
    # Control: identical population keeps the delta column (all +0 here).
    rep_a = _probe_report(tmp_path, "alpha", 40, 20)
    md = to_markdown(rep_a, rep_a)
    assert "different populations" not in md
    assert "+0.0000" in md
    table = _headline_table_lines(md)
    assert all(not ln.endswith("| — |") for ln in table), table


def test_predictor_mode_recorded_in_report(tmp_path):
    # Ad-hoc scorers keep working (None); the protocol stays name + score.
    assert _probe_report(tmp_path, "nomode", 40, 20)["predictor_mode"] is None
    rep = _probe_report(tmp_path, "withmode", 40, 20, mode="soft_vote")
    assert rep["predictor_mode"] == "soft_vote"
    assert rep["schema_version"] == "1.4.0"


def test_vs_line_attributes_baseline_contract(tmp_path):
    # Same bytes, different contracts: deltas kept (valid system comparison)
    # but both sides named so nothing is silently attributed.
    rep_a = _probe_report(tmp_path, "gamma", 40, 20, mode="vote_fraction")
    rep_b = {**rep_a, "predictor": "candidate", "predictor_mode": "soft_vote"}
    md = to_markdown(rep_a, rep_b)
    assert "vs baseline `candidate` (mode `soft_vote`)" in md
    assert "mix every pipeline difference" in md
    assert "different populations" not in md
    assert "+0.0000" in md

    # Same name, provably different mode: still attributed.
    rep_c = {**rep_a, "predictor_mode": "soft_vote"}
    assert "vs baseline `probe` (mode `soft_vote`)" in to_markdown(rep_a, rep_c)

    # Self-compare and unknown modes: silent (nothing to attribute,
    # and missing mode proves nothing).
    assert "vs baseline `" not in to_markdown(rep_a, rep_a)
    plain = _probe_report(tmp_path, "delta", 40, 20)
    assert "vs baseline `" not in to_markdown(plain, plain)


def test_mismatch_note_names_both_contracts(tmp_path):
    rep_a = _probe_report(tmp_path, "eps", 40, 20, mode="soft_vote")
    rep_b = _probe_report(tmp_path, "zeta", 30, 30, mode="vote_fraction")
    md = to_markdown(rep_a, {**rep_b, "predictor": "old-baseline"})
    assert "different populations" in md
    assert "`old-baseline` (mode `vote_fraction`)" in md
    assert "`probe` (mode `soft_vote`)" in md
