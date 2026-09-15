"""Tests for the mechanism hard gates + shape advisory in validate_cc_benign.

Gate spec under test (two-number form):
  hard: path-depth single-feature ROC-AUC two-sided |AUC - 0.5|
        <= PATH_DEPTH_AUC_MAXDIST (0.05); URL-length inversion
        mean(benign) - mean(phish) >= URL_LEN_INVERSION_MIN (0.0).
  advisory: trial-split total shape AUC vs SHAPE_AUC_ADVISORY (0.70) —
        reported, never fails.

The two-sided depth form is load-bearing: the frozen 0.753-era splits
carry path_depth AUC ~0.30 (benign deeper — inverted), which a one-sided
"<= 0.55" gate scores as a pass. All fixtures are synthetic and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import validate_cc_benign as V


def _write_jsonl(path: Path, urls: list[str], label: int) -> None:
    rows = [{"url": u, "label": label} for u in urls]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _write_split(
    path: Path,
    train_urls: list[str],
    train_labels: list[int],
    test_urls: list[str],
    test_labels: list[int],
) -> None:
    import pandas as pd

    def frame(urls: list[str], labels: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "url": urls,
                "label": labels,
                "first_seen": ["2026-08-01T00:00:00+00:00"] * len(urls),
                "registrable_domain": [f"d{i}.example.com" for i in range(len(urls))],
                "suffix": ["com"] * len(urls),
                "source": ["synthetic"] * len(urls),
            }
        )

    te = frame(test_urls, test_labels)
    # Train and test must be eTLD+1-disjoint (else the overlap hard gate,
    # not the advisory, decides the exit code).
    te["registrable_domain"] = [f"t{i}.example.net" for i in range(len(test_urls))]
    tr = frame(train_urls, train_labels)
    tr["registrable_domain"] = [f"d{i}.example.com" for i in range(len(train_urls))]
    (path / "train.csv").write_text(tr.to_csv(index=False), encoding="utf-8")
    (path / "test.csv").write_text(te.to_csv(index=False), encoding="utf-8")


def test_path_depth_gate_is_two_sided() -> None:
    """Frozen-era direction (benign deeper, AUC ~0.3) must still fire."""
    auc = V.single_feature_auc([3.0] * 100, [1.0] * 100)
    assert auc < 0.5
    assert abs(auc - 0.5) > V.PATH_DEPTH_AUC_MAXDIST


def test_depth_gate_fires_end_to_end(tmp_path: Path) -> None:
    """Roots vs depth-4 paths: AUC 1.0, dist 0.5, breach (rc 1)."""
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, [f"http://b{i}.example.com/" for i in range(200)], 0)
    _write_jsonl(phish, [f"http://p{i}.example.net/a/b/c/d" for i in range(200)], 1)
    report = tmp_path / "report.json"
    rc = V.main(
        ["--benign", str(benign), "--phish-glob", str(phish), "--out", str(report)]
    )
    assert rc == 1
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert any(f.startswith("path_depth_auc_dist=") for f in rep["failures"])
    assert rep["mechanism_gates"]["path_depth_gate_passed"] is False


def test_inversion_gate_fires_on_short_benign(tmp_path: Path) -> None:
    """Same depth both sides, benign shorter: inversion breach only."""
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, [f"http://b{i}.example.com/" for i in range(200)], 0)
    _write_jsonl(
        phish, [f"http://very-long-hostname-{i}.example.net/" for i in range(200)], 1
    )
    report = tmp_path / "report.json"
    rc = V.main(
        ["--benign", str(benign), "--phish-glob", str(phish), "--out", str(report)]
    )
    assert rc == 1
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert any(f.startswith("url_len_inversion=") for f in rep["failures"])
    assert rep["mechanism_gates"]["url_len_gate_passed"] is False
    assert rep["mechanism_gates"]["path_depth_gate_passed"] is True


def test_mechanism_gates_pass_on_matched(tmp_path: Path) -> None:
    """Same shapes both sides: all mechanism gates pass (rc 0)."""
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, [f"http://host{i}.example.com/a/b?q=1" for i in range(200)], 0)
    _write_jsonl(phish, [f"http://host{i}.example.net/a/b?q=1" for i in range(200)], 1)
    report = tmp_path / "report.json"
    rc = V.main(
        ["--benign", str(benign), "--phish-glob", str(phish), "--out", str(report)]
    )
    assert rc == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["mechanism_gates"]["path_depth_gate_passed"] is True
    assert rep["mechanism_gates"]["url_len_gate_passed"] is True


def test_shape_advisory_trips_without_failing(tmp_path: Path) -> None:
    """Separable trial split: advisory tripped True, rc still 0."""
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, [f"http://b{i}.example.com/" for i in range(50)], 0)
    _write_jsonl(phish, [f"http://p{i}.example.net/" for i in range(50)], 1)
    split = tmp_path / "split"
    split.mkdir()
    train_urls = [f"http://b{i}.example.com/" for i in range(50)]
    train_urls += [f"http://p{i}.example.net/a/b/c/d/e/f" for i in range(50)]
    test_urls = [f"http://q{i}.example.org/" for i in range(50)]
    test_urls += [f"http://r{i}.example.io/a/b/c/d/e/f" for i in range(50)]
    _write_split(split, train_urls, [0] * 50 + [1] * 50, test_urls, [0] * 50 + [1] * 50)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(benign),
            "--phish-glob",
            str(phish),
            "--split-dir",
            str(split),
            "--out",
            str(report),
        ]
    )
    assert rc == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["shape_advisory"]["tripped"] is True
    assert rep["shape_advisory"]["threshold"] == V.SHAPE_AUC_ADVISORY
    assert not any(f.startswith("shape") for f in rep["failures"])


def test_shape_advisory_quiet_on_parity(tmp_path: Path) -> None:
    """Matched trial split: advisory present and untripped."""
    benign = tmp_path / "b.jsonl"
    phish = tmp_path / "p.jsonl"
    _write_jsonl(benign, [f"http://host{i}.example.com/a/b?q=1" for i in range(50)], 0)
    _write_jsonl(phish, [f"http://host{i}.example.net/a/b?q=1" for i in range(50)], 1)
    split = tmp_path / "split"
    split.mkdir()
    train_urls = [f"http://h{i}.example.com/a/b?q=1" for i in range(50)]
    train_urls += [f"http://h{i}.example.net/a/b?q=1" for i in range(50)]
    test_urls = [f"http://k{i}.example.org/a/b?q=1" for i in range(50)]
    test_urls += [f"http://k{i}.example.io/a/b?q=1" for i in range(50)]
    _write_split(split, train_urls, [0] * 50 + [1] * 50, test_urls, [0] * 50 + [1] * 50)
    report = tmp_path / "report.json"
    rc = V.main(
        [
            "--benign",
            str(benign),
            "--phish-glob",
            str(phish),
            "--split-dir",
            str(split),
            "--out",
            str(report),
        ]
    )
    assert rc == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["shape_advisory"]["tripped"] is False
