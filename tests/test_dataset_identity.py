"""Golden tests for the frozen Phase 1 dataset identity.

These lock the reproducibility guarantees that earlier audits found to be
implicit: the baseline's dataset hash, the canonical CRLF worktree bytes it
is defined on, manifest/count agreement, manifest input presence, leakage
correspondence, domain disjointness, and phishing temporal purity.

These tests pin the frozen Phase 1 and Phase 2 CSV identities (waived
instruments, see docs/WAIVERS.md); the successor population is pinned by
``repro/hashes.json`` and checked by ``repro/verify.py``. All files
referenced here are git-tracked.
"""

from __future__ import annotations

import hashlib
import json
import socket as _socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pytest

import build_splits

pytestmark = pytest.mark.golden

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
BASELINE = ROOT / "reports" / "baseline.json"

# Waived instruments (see docs/WAIVERS.md): these CSVs can never be rebuilt,
# so a pinned sha256 is the only thing defending them. Phase 1 test.csv is
# additionally cross-checked against reports/baseline.json below.
FROZEN_CSVS = {
    "data/splits/train.csv": (
        "d2066b0ccdc88112f4acc812b675a994b39fe847daccffd0d3043a4f49284427"
    ),
    "data/splits/test.csv": (
        "385aa409c222247f04255f760fa3e86ee9e38d3d6d08b1d410ff57f6780b49fb"
    ),
    "data/splits-large/train.csv": (
        "c2a85580df5dc98c0516706a3526e0811332e1c3d436709bfc90f3db6d8bb4eb"
    ),
    "data/splits-large/test.csv": (
        "961efe5c9bfc95495446223e92d9ddfb50bf85c4888a7fa6ac83cb510260c379"
    ),
}


class _BlockedSocket(_socket.socket):
    """A socket that refuses to connect: any live fetch fails loudly."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("network access blocked in golden tests")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any live fetch fails loudly instead of silently drifting the suite.

    The builder is pinned to the bundled PSL snapshot, so nothing here
    should touch the network. Patch with a subclass (never a plain
    function): replacing socket.socket outright breaks modules that
    subclass it at import time.
    """
    monkeypatch.setattr(_socket, "socket", _BlockedSocket)


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


@pytest.mark.parametrize(("relpath", "expected"), sorted(FROZEN_CSVS.items()))
def test_frozen_csv_sha_pinned(relpath: str, expected: str) -> None:
    """Waived instruments are defended by pinned hashes, nothing else."""
    assert _sha256(ROOT / relpath) == expected, relpath


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
    """The recorded audit is a tight function of the committed CSVs.

    Float fields compare with pytest.approx, not ==: the audit fits an
    iterative optimizer (lbfgs), whose last-ulp scores legitimately differ
    between BLAS builds (Windows vs Linux CI) while meaning nothing. The
    tolerance (rel=1e-9) sits orders of magnitude above ulp noise and
    orders below any real drift — the known 7-row rebuild gap moved ROC in
    the third decimal. Verdicts compare exactly.
    """
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
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
        else:
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


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        # Snapshot-era grouping: these suffixes postdate the bundled
        # snapshot, so hosts collapse to the parent domain. A live PSL
        # would group per-entity (e.g. x.pages.dev) and reshuffle the
        # campaign caps — this table fails visibly on any such swap.
        ("x.pages.dev", "pages.dev"),
        ("x.vercel.app", "vercel.app"),
        ("x.netlify.app", "netlify.app"),
        ("x.gitbook.io", "gitbook.io"),
        # Controls: stable under any PSL vintage.
        ("about.gitlab.com", "gitlab.com"),
        ("example.co.uk", "example.co.uk"),
    ],
)
def test_shared_suffix_hosts_group_by_snapshot(host: str, expected: str) -> None:
    e = build_splits.EXTRACT(host)
    assert (f"{e.domain}.{e.suffix}" if e.suffix else e.domain) == expected
