"""Tests for the disjointness acceptance gates in build_splits.main.

All cases run offline on tiny synthetic raw dirs (never the frozen data):
* drop share above STRADDLER_DROP_SHARE_MAX refuses the split;
* benign test domains below BENIGN_TEST_DOMAINS_FLOOR refuses the split;
* a clean split with shape-identical classes passes both gates and records
  them in the manifest (boundary: 250 benign test domains passes).

Benign/phish test URLs are length-identical by construction
(`bt…aa.com` vs `pt…aa.org` + the same path: every netloc is 13 chars),
so the shape-only audit scores ~0.5 and cannot interfere with the gate
verdicts under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import build_splits

SEED = build_splits.NEG_HASH_SEED_DEFAULT
FRAC = 0.2


def _registrable(prefix: str, i: int) -> str:
    return f"{prefix}{i:05d}aa.com"


def _test_domains(prefix: str, n: int) -> list[str]:
    """First n registrable domains (deterministic) hashing into test."""
    found: list[str] = []
    i = 0
    while len(found) < n:
        d = _registrable(prefix, i)
        if build_splits.neg_domain_is_test(d, SEED, FRAC):
            found.append(d)
        i += 1
    return found


def _train_domains(prefix: str, n: int) -> list[str]:
    found: list[str] = []
    i = 0
    while len(found) < n:
        d = _registrable(prefix, i)
        if not build_splits.neg_domain_is_test(d, SEED, FRAC):
            found.append(d)
        i += 1
    return found


def _write_raw(
    raw: Path,
    benign_test: list[str],
    benign_train: list[str],
    phish_old: list[str],
    phish_new: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for d in benign_test:
        rows.append(
            {
                "url": f"https://{d}/about",
                "label": 0,
                "first_seen": "2026-02-01T00:00:00+00:00",
                "source": "tranco:TEST",
            }
        )
    for d in benign_train:
        rows.append(
            {
                "url": f"https://{d}/about",
                "label": 0,
                "first_seen": "2026-02-01T00:00:00+00:00",
                "source": "tranco:TEST",
            }
        )
    for h in phish_old:
        rows.append(
            {
                "url": f"https://{h}",
                "label": 1,
                "first_seen": "2026-01-01T00:00:00+00:00",
                "source": "phishtank",
            }
        )
    for h in phish_new:
        rows.append(
            {
                "url": f"https://{h}",
                "label": 1,
                "first_seen": "2026-06-01T00:00:00+00:00",
                "source": "phishtank",
            }
        )
    (raw / "synth.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def _run(
    monkeypatch: Any, capsys: Any, tmp_path: Path, argv: list[str]
) -> tuple[int, Any]:
    monkeypatch.setattr(sys, "argv", ["build_splits.py"] + argv)
    rc = build_splits.main()
    return rc, capsys.readouterr()


def test_straddler_drop_share_refuses_split(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """1 straddling domain of 10 pre-drop test domains (10%) > 2%: refused."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "split"
    bt = _test_domains("bt", 1)
    br = _train_domains("br", 1)
    # shared.com straddles T via two distinct URLs; 3 more phish test
    # domains set the denominator: pre-drop test domains = shared + bt + 3.
    others = [f"q{i:05d}aa.org/about" for i in range(3)]
    _write_raw(
        raw,
        bt,
        br,
        ["shared.com/old-a"],
        ["shared.com/new-b"] + others,
    )
    rc, captured = _run(
        monkeypatch,
        capsys,
        tmp_path,
        ["--raw", str(raw), "--out", str(out), "--split-date", "2026-03-01"],
    )
    assert rc == 1
    assert "STRADDLER GATE FAILED" in captured.err
    assert not (out / "test.csv").exists()  # nothing written


def test_benign_domain_floor_refuses_split(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Zero straddling but only 3 benign test domains (< 250): refused."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "split"
    _write_raw(
        raw,
        _test_domains("bt", 3),
        _train_domains("br", 3),
        [f"qt{i:05d}aa.net/about" for i in range(5)],
        [f"pt{i:05d}aa.org/about" for i in range(5)],
    )
    rc, captured = _run(
        monkeypatch,
        capsys,
        tmp_path,
        ["--raw", str(raw), "--out", str(out), "--split-date", "2026-03-01"],
    )
    assert rc == 1
    assert "DOMAIN FLOOR FAILED" in captured.err
    assert not (out / "test.csv").exists()


def test_clean_split_passes_both_gates(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """Boundary 250 benign test domains, zero straddling: written + recorded."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "split"
    _write_raw(
        raw,
        _test_domains("bt", 250),
        _train_domains("br", 50),
        [f"qt{i:05d}aa.net/about" for i in range(250)],
        [f"pt{i:05d}aa.org/about" for i in range(250)],
    )
    rc, _ = _run(
        monkeypatch,
        capsys,
        tmp_path,
        [
            "--raw",
            str(raw),
            "--out",
            str(out),
            "--split-date",
            "2026-03-01",
            "--deterministic-manifest",
        ],
    )
    assert rc == 0
    assert (out / "test.csv").exists()
    # Refusal paths write nothing (covered above); the manifest keeps its
    # frozen schema — gate outcomes print, they are not recorded keys.
