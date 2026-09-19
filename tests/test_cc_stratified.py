"""Tests for the stratified shape gate (Amendment D, D0.2).

* --mode defaults to unstratified with byte-identical reports (no
  "stratified" key);
* stratified mode gates main-benign vs non-hosted phishing (blocking)
  and reports hosted vs hosted (descriptive, never fails);
* the unstratified block is retained in every stratified report;
* regression: stratified mode on the pinned 12k regenerates the numbers
  in reports/m1-stratified.json (skipped where the corpus is absent —
  data/ is git-ignored and not present in CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import validate_cc_benign as V

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "raw" / "benign-cc-CC-MAIN-2026-34-2026-09-15.jsonl"
M1_REPORT = ROOT / "reports" / "m1-stratified.json"
D01_DAYS = ("12", "13", "14", "15", "16")
D01_FILES = sorted(
    f"{src}-2026-09-{day}.jsonl"
    for src in ("openphish", "phishtank")
    for day in D01_DAYS
)


def _write_jsonl(path: Path, urls: list[str], label: int) -> None:
    rows = [{"url": u, "label": label} for u in urls]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _mixed(host: str, suffix: str, n: int) -> list[str]:
    """25/41/16/18 root/path1/pathN/query mix with matched shapes."""
    urls = [f"http://r{i}.{host}.{suffix}/" for i in range(25)]
    urls += [f"http://p{i}.{host}.{suffix}/a" for i in range(41)]
    urls += [f"http://q{i}.{host}.{suffix}/a/b" for i in range(16)]
    urls += [f"http://s{i}.{host}.{suffix}/a?x=1" for i in range(18)]
    return urls[:n]


def test_default_mode_has_no_stratified_key(tmp_path: Path) -> None:
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, _mixed("b", "example.com", 100), 0)
    _write_jsonl(phish, _mixed("p", "example.net", 100), 1)
    report = tmp_path / "report.json"
    rc = V.main(
        ["--benign", str(benign), "--phish-glob", str(phish), "--out", str(report)]
    )
    assert rc == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert "stratified" not in rep


def test_stratified_main_passes_on_matched(tmp_path: Path) -> None:
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, _mixed("b", "example.com", 100), 0)
    _write_jsonl(phish, _mixed("p", "example.net", 100), 1)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(benign),
            "--phish-glob",
            str(phish),
            "--mode",
            "stratified",
            "--out",
            str(report),
        ]
    )
    assert rc == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    main = rep["stratified"]["main"]
    assert main["failures"] == []
    assert main["n_benign"] == 100 and main["n_phish"] == 100
    assert rep["stratified"]["hosted"]["n_benign"] == 0
    # Unstratified block retained beside the stratified verdict.
    assert rep["failures"] == []
    assert rep["type_drift"]["root"] == pytest.approx(0.0)


def test_stratified_main_failure_blocks(tmp_path: Path) -> None:
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    roots = [f"http://b{i}.example.com/" for i in range(60)]
    hosted_benign = [f"https://tenant{i}.vercel.app/a/b/c" for i in range(6)]
    _write_jsonl(benign, roots + hosted_benign, 0)
    _write_jsonl(phish, _mixed("p", "example.net", 100), 1)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(benign),
            "--phish-glob",
            str(phish),
            "--mode",
            "stratified",
            "--out",
            str(report),
        ]
    )
    assert rc == 1
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert any(f.startswith("main:type_drift[root]=") for f in rep["failures"])
    assert not any(f.startswith("hosted:") for f in rep["failures"])
    host = rep["stratified"]["hosted"]
    assert host["n_benign"] == 6 and host["n_phish"] == 0
    assert host["failures"] == []  # descriptive, however mismatched


def test_hosted_stratum_never_fails() -> None:
    """Wildly mismatched hosted sides: reported, zero failures."""
    benign = [f"https://tenant{i}.vercel.app/a/b/c/d" for i in range(20)]
    phish = [f"https://evil{i}.vercel.app/" for i in range(20)]
    m = V.stratum_metrics(benign, phish, blocking=False)
    assert m["type_drift"]["root"] == pytest.approx(1.0)
    assert m["failures"] == []
    m2 = V.stratum_metrics([], phish, blocking=False)
    assert m2["path_depth_auc"] is None and m2["url_len_inversion"] is None
    assert m2["failures"] == []


@pytest.mark.skipif(
    not CORPUS.exists(), reason="pinned 12k corpus absent (data/ git-ignored)"
)
def test_m1_stratified_regression(tmp_path: Path) -> None:
    """Stratified mode regenerates reports/m1-stratified.json exactly."""
    expected = json.loads(M1_REPORT.read_text(encoding="utf-8"))
    phish_glob = " ".join(str(ROOT / "data" / "raw" / f) for f in D01_FILES)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(CORPUS),
            "--phish-glob",
            phish_glob,
            "--mode",
            "stratified",
            "--out",
            str(report),
        ]
    )
    assert rc == 1  # D3 unavailable: main stratum fails
    rep = json.loads(report.read_text(encoding="utf-8"))
    main = rep["stratified"]["main"]
    assert main["n_benign"] == expected["n_benign"] == 11795
    assert main["n_phish"] == expected["n_phish_nonhosted"] == 60880
    assert rep["stratified"]["hosted"]["n_benign"] == expected["n_benign_hosted"] == 205
    assert rep["stratified"]["hosted"]["n_phish"] == expected["n_phish_hosted"] == 16221
    for t in ("root", "path1", "pathN", "query"):
        assert main["type_drift"][t] == pytest.approx(
            expected["type_drift"][t], rel=1e-9
        )
    assert main["scheme_gap"] == pytest.approx(expected["scheme_gap"], rel=1e-9)
    assert main["path_depth_auc_dist"] == pytest.approx(
        expected["path_depth_auc_dist"], rel=1e-9
    )
    assert main["url_len_inversion"] == pytest.approx(
        expected["url_len_inversion"], rel=1e-9
    )
    # Registered D0.2 names on the report section.
    for key in ("type_drift", "scheme_gap", "path_depth_auc_dist", "url_len_inversion"):
        assert key in main
    # Six main failures, nothing else: the 12k passes unstratified.
    assert len(rep["failures"]) == 6
    assert all(f.startswith("main:") for f in rep["failures"])
    assert any(f.startswith("main:type_drift[root]=") for f in rep["failures"])
