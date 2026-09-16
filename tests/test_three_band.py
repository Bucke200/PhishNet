"""Three-band split tests (offline fixtures, never the frozen data).

Covers what the two-band gates cannot: phishing lands in train/calib/test
by era (T1/T2), benign domains land by hash bucket, straddlers drop from
later bands only, misconfigurations refuse before writing, and two runs
produce identical bytes. Shape-identical URLs throughout (13-char
netlocs, same depth/scheme), so the shape audit cannot interfere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import build_splits

SEED = build_splits.NEG_HASH_SEED_DEFAULT
TEST_FRAC = 0.3
CALIB_FRAC = 0.2
T1 = "2026-07-25"
T2 = "2026-08-22"

BASE_ARGV = [
    "--calib-date",
    T1,
    "--split-date",
    f"{T2}T00:00:00+00:00",
    "--benign-test-fraction",
    str(TEST_FRAC),
    "--benign-calib-fraction",
    str(CALIB_FRAC),
    "--min-benign-test-domains",
    "1",
    "--min-benign-calib-domains",
    "1",
    "--max-straddler-drop-share",
    "1.0",
    "--deterministic-manifest",
]


def _bucket_domains(bucket: str, n: int) -> list[str]:
    """First n registrable domains (deterministic) hashing into a bucket."""
    found: list[str] = []
    i = 0
    while len(found) < n:
        d = f"bb{i:05d}aa.com"
        if build_splits.benign_bucket(d, SEED, TEST_FRAC, CALIB_FRAC) == bucket:
            found.append(d)
        i += 1
    return found


def _write_raw(
    raw: Path,
    benign: list[str],
    phish: list[tuple[str, str]],
) -> None:
    # Shape-identical by construction (13-char netlocs, same path): the
    # shape-only audit must not fire on fixtures.
    rows: list[dict[str, Any]] = [
        {
            "url": f"https://{d}/x",
            "label": 0,
            "first_seen": "2026-02-01T00:00:00+00:00",
            "source": "tranco:TEST",
        }
        for d in benign
    ]
    rows += [
        {
            "url": f"https://{h}/x",
            "label": 1,
            "first_seen": stamp,
            "source": "phishtank",
        }
        for h, stamp in phish
    ]
    (raw / "synth.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def _run(
    monkeypatch: Any,
    tmp_path: Path,
    raw_argv: list[str],
    tag: str,
    rows: tuple[list[str], list[tuple[str, str]]],
) -> tuple[int, Path]:
    raw = tmp_path / f"raw-{tag}"
    out = tmp_path / f"out-{tag}"
    raw.mkdir()
    _write_raw(raw, rows[0], rows[1])
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_splits.py", *raw_argv, "--raw", str(raw), "--out", str(out)],
    )
    return build_splits.main(), out


def _standard_rows() -> tuple[list[str], list[tuple[str, str]]]:
    benign = (
        _bucket_domains("test", 3)
        + _bucket_domains("calib", 3)
        + _bucket_domains("train", 3)
    )
    phish = [
        ("pa00001aa.com", "2026-01-01T00:00:00+00:00"),  # train era
        ("pa00002aa.com", "2026-01-02T00:00:00+00:00"),
        ("pb00001aa.com", "2026-08-01T00:00:00+00:00"),  # calib era
        ("pb00002aa.com", "2026-08-02T00:00:00+00:00"),
        ("pc00001aa.com", "2026-09-01T00:00:00+00:00"),  # test era
        ("pc00002aa.com", "2026-09-02T00:00:00+00:00"),
    ]
    return benign, phish


def test_bands_split_by_era_and_bucket(monkeypatch: Any, tmp_path: Path) -> None:
    rc, out = _run(monkeypatch, tmp_path, BASE_ARGV, "bands", _standard_rows())
    assert rc == 0
    train = pd.read_csv(out / "train.csv")
    calib = pd.read_csv(out / "calib.csv")
    test = pd.read_csv(out / "test.csv")
    assert set(train[train.label == 1]["url"].str.contains("pa")) == {True}
    assert set(calib[calib.label == 1]["url"].str.contains("pb")) == {True}
    assert set(test[test.label == 1]["url"].str.contains("pc")) == {True}
    for frame, bucket in ((train, "train"), (calib, "calib"), (test, "test")):
        ben = frame[frame.label == 0]["registrable_domain"].unique()
        assert ben.size > 0
        for d in ben:
            assert build_splits.benign_bucket(d, SEED, TEST_FRAC, CALIB_FRAC) == bucket
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["bands"]["t1_calib_date"].startswith(T1)
    assert manifest["bands"]["t2_test_cutoff"].startswith(T2)
    assert manifest["n_calib"] == len(calib)
    assert manifest["n_calib_phish"] == 2 and manifest["n_calib_benign"] == 3
    assert manifest["benign_split"]["calib_fraction"] == CALIB_FRAC
    assert manifest["calib_straddling_domains_dropped"] == 0
    assert manifest["calib_leakage_audit"]["verdict"] in ("ok", "suspicious")


def test_straddlers_drop_from_later_bands_only(
    monkeypatch: Any, tmp_path: Path
) -> None:
    benign, phish = _standard_rows()
    # One kit domain across all three eras: survives only in train.
    # Same-length paths keep the shape audit quiet.
    phish += [
        ("sh00001aa.com/w", "2026-01-03T00:00:00+00:00"),
        ("sh00001aa.com/x", "2026-08-03T00:00:00+00:00"),
        ("sh00001aa.com/y", "2026-09-03T00:00:00+00:00"),
    ]
    rc, out = _run(monkeypatch, tmp_path, BASE_ARGV, "straddle", (benign, phish))
    assert rc == 0
    train = pd.read_csv(out / "train.csv")
    calib = pd.read_csv(out / "calib.csv")
    test = pd.read_csv(out / "test.csv")
    assert "sh00001aa.com" in set(train["registrable_domain"])
    assert "sh00001aa.com" not in set(calib["registrable_domain"])
    assert "sh00001aa.com" not in set(test["registrable_domain"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["calib_straddling_domains_dropped"] >= 1
    assert manifest["straddling_domains_dropped"] >= 1
    # Full disjointness across all three bands.
    assert not (
        set(train["registrable_domain"])
        & set(calib["registrable_domain"])
        & set(test["registrable_domain"])
    )
    assert not (set(train["registrable_domain"]) & set(calib["registrable_domain"]))
    assert not (set(train["registrable_domain"]) & set(test["registrable_domain"]))
    assert not (set(calib["registrable_domain"]) & set(test["registrable_domain"]))


def test_phase3_tenant_grouping_keeps_hosted_tenants(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Hosted tenants are separate attackers: distinct tenants in different
    eras all survive (no platform-level straddler drop), the same tenant
    across eras drops from the later band, and is_hosted_tenant rides the
    CSVs for the hosted eval slice."""
    benign, phish = _standard_rows()
    phish += [
        ("tenanta.blogspot.com/x", "2026-01-03T00:00:00+00:00"),
        ("tenantb.blogspot.com/x", "2026-08-03T00:00:00+00:00"),
        ("tenantc.blogspot.com/x", "2026-09-03T00:00:00+00:00"),
        ("reused.blogspot.com/x", "2026-01-04T00:00:00+00:00"),
        ("reused.blogspot.com/y", "2026-09-04T00:00:00+00:00"),
    ]
    argv = [*BASE_ARGV, "--phase3"]
    rc, out = _run(monkeypatch, tmp_path, argv, "tenants", (benign, phish))
    assert rc == 0
    train = pd.read_csv(out / "train.csv")
    calib = pd.read_csv(out / "calib.csv")
    test = pd.read_csv(out / "test.csv")
    assert "split_group" in train.columns
    assert "is_hosted_tenant" in test.columns
    assert "tenanta.blogspot.com" in set(train["split_group"])
    assert "tenantb.blogspot.com" in set(calib["split_group"])
    assert "tenantc.blogspot.com" in set(test["split_group"])
    assert "reused.blogspot.com" in set(train["split_group"])
    assert "reused.blogspot.com" not in set(test["split_group"])
    hosted_test = test[test["is_hosted_tenant"].astype(str) == "True"]
    assert len(hosted_test) == 1  # the tenant survives into test now
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["host_grouping"]["test"]["n_hosted"] == 1


def test_misconfigurations_refuse(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    rows = _standard_rows()
    raw = tmp_path / "raw-ref"
    out = tmp_path / "out-ref"
    raw.mkdir()
    _write_raw(raw, rows[0], rows[1])
    cases = [
        # calib date without a calib fraction: calib would hold phish only.
        ["--calib-date", T1, "--split-date", f"{T2}T00:00:00+00:00"],
        # calib fraction without a calib date: phish calib band undefined.
        ["--split-date", f"{T2}T00:00:00+00:00", "--benign-calib-fraction", "0.2"],
        # T1 after T2.
        [
            "--calib-date",
            "2026-09-01",
            "--split-date",
            f"{T2}T00:00:00+00:00",
            "--benign-calib-fraction",
            "0.2",
        ],
        # Fractions leave no train bucket.
        [
            "--calib-date",
            T1,
            "--split-date",
            f"{T2}T00:00:00+00:00",
            "--benign-test-fraction",
            "0.8",
            "--benign-calib-fraction",
            "0.2",
        ],
    ]
    for argv in cases:
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_splits.py", *argv, "--raw", str(raw), "--out", str(out)],
        )
        with __import__("pytest").raises(SystemExit):
            build_splits.main()


def test_three_band_determinism(monkeypatch: Any, tmp_path: Path) -> None:
    rows = _standard_rows()
    _, out1 = _run(monkeypatch, tmp_path, BASE_ARGV, "det1", rows)
    _, out2 = _run(monkeypatch, tmp_path, BASE_ARGV, "det2", rows)
    for name in ("train.csv", "calib.csv", "test.csv", "manifest.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
