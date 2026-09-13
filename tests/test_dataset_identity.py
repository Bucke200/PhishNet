"""Golden tests for the frozen Phase 1 dataset identity.

These lock the reproducibility guarantees that earlier audits found to be
implicit: the baseline's dataset hash, the canonical CRLF worktree bytes it
is defined on, manifest/count agreement, manifest input presence, leakage
correspondence, domain disjointness, and phishing temporal purity.

All files referenced here are git-tracked. Worktree-only populations
(``data/splits-large/``, ``data/splits-eval/``) are intentionally excluded:
they cannot be required of a fresh clone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pytest

pytestmark = pytest.mark.golden

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
BASELINE = ROOT / "reports" / "baseline.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> dict[str, Any]:
    return dict(json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8")))


def test_dataset_sha_matches_baseline_identity() -> None:
    """The frozen identity: test.csv bytes hash to the recorded value.

    This fails on checkouts with LF line endings (the reported
    Windows-vs-Linux divergence); .gitattributes forces CRLF everywhere.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert _sha256(SPLITS / "test.csv") == baseline["dataset"]["sha256"]


def test_hashed_files_are_canonical_crlf() -> None:
    """Every newline in the hashed files is part of a CRLF pair."""
    for path in (
        SPLITS / "train.csv",
        SPLITS / "test.csv",
        ROOT / "reports" / "baseline.json",
        SPLITS / "manifest.json",
    ):
        raw = path.read_bytes()
        assert raw.count(b"\n") == raw.count(b"\r\n"), path


def test_no_lone_lf_in_data_or_reports() -> None:
    """The committed eol=crlf rule must hold for all of data/ and reports/.

    A single LF-normalized file breaks byte-identical hashes on other
    platforms, so this scans every file rather than a pinned subset.
    """
    offenders = []
    for base in (ROOT / "data", ROOT / "reports"):
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if raw.count(b"\n") != raw.count(b"\r\n"):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_manifest_reconciles_with_splits() -> None:
    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    man = _manifest()
    assert man["n_train"] == len(train)
    assert man["n_test"] == len(test)
    assert man["n_train"] == man["n_train_phish"] + man["n_train_benign"]
    assert man["n_test"] == man["n_test_phish"] + man["n_test_benign"]
    assert int((train.label == 1).sum()) == man["n_train_phish"]
    assert int((train.label == 0).sum()) == man["n_train_benign"]
    assert int((test.label == 1).sum()) == man["n_test_phish"]
    assert int((test.label == 0).sum()) == man["n_test_benign"]


def test_manifest_inputs_exist() -> None:
    man = _manifest()
    raw_files = man["raw_files"]
    assert isinstance(raw_files, list) and raw_files
    for name in raw_files:
        assert (ROOT / "data" / "raw" / str(name)).is_file(), name


def test_leakage_audit_matches_committed_data() -> None:
    """The recorded audit is an exact function of the committed CSVs."""
    import build_splits

    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    for frame in (train, test):
        frame["path_depth"] = frame["url"].map(
            lambda u: len([s for s in urlparse(str(u)).path.split("/") if s])
        )
    got = build_splits.leakage_audit(train, test)
    recorded = _manifest()["leakage_audit"]
    assert isinstance(recorded, dict)
    for key, value in recorded.items():
        assert got[key] == value, key


def test_no_domain_overlap_and_temporal_purity() -> None:
    train = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    assert not (set(train["registrable_domain"]) & set(test["registrable_domain"]))
    cutoff = pd.Timestamp(str(_manifest()["split_date"]), tz="UTC")
    train_seen = pd.to_datetime(train["first_seen"], utc=True, format="mixed")
    test_seen = pd.to_datetime(test["first_seen"], utc=True, format="mixed")
    assert bool((train_seen[train.label == 1] < cutoff).all())
    assert bool((test_seen[test.label == 1] >= cutoff).all())
