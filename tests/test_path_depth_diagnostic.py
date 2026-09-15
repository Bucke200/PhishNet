"""Tests for the netloc_len pre-match diagnostic.

The diagnostic lives in ``scratch/path_depth_diagnostic.py`` (a
non-packaged script, hence loaded by file path). These tests pin two
properties:

* ``netloc_len_values`` delegates to the existing
  ``build_splits.shape_features`` implementation (column 1) instead of
  reimplementing netloc parsing.
* ``run_diagnostic`` reports pre-match statistics over the FULL benign
  test population (before stratified-subsampling matching), with the
  matched-subset statistics kept separate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_splits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_diagnostic() -> Any:
    path = REPO_ROOT / "scratch" / "path_depth_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "path_depth_diagnostic", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["path_depth_diagnostic"] = module
    spec.loader.exec_module(module)
    return module


diag = _load_diagnostic()


def test_netloc_len_delegates_to_shape_features(
    monkeypatch: Any,
) -> None:
    urls = pd.Series(["https://www.example.com/path", "https://b.co/"])
    got = diag.netloc_len_values(urls)
    expected = build_splits.shape_features(pd.DataFrame({"url": list(urls)}))[
        :, diag.NETLOC_LEN_INDEX
    ]
    assert np.array_equal(got, expected)

    calls: list[str] = []

    def _fake(frame: pd.DataFrame) -> np.ndarray:
        calls.append("shape_features")
        return np.array([[1.0, 7.0], [2.0, 9.0]])

    monkeypatch.setattr(build_splits, "shape_features", _fake)
    assert list(diag.netloc_len_values(urls)) == [7.0, 9.0]
    assert calls == ["shape_features"]


def _write_eval_dir(base: Path) -> tuple[int, int]:
    """Tiny synthetic eval population where matching must discard rows."""
    test_rows = [
        # phish: 4x depth-0, 1x depth-1, 1x depth-2
        ("https://phish-a.com/", 1, "phish-a.com"),
        ("https://phish-b.com/", 1, "phish-b.com"),
        ("https://phish-c.com/", 1, "phish-c.com"),
        ("https://phish-d.com/", 1, "phish-d.com"),
        ("https://phish-e.com/about", 1, "phish-e.com"),
        ("https://phish-f.com/a/b", 1, "phish-f.com"),
        # benign: 6x depth-0 (long netlocs), 3x depth-1, 3x depth-2
        *[
            (
                f"https://averylongbarehomepagehostname{i:02d}.example.com/",
                0,
                f"long{i:02d}.example.com",
            )
            for i in range(6)
        ],
        ("https://s1.com/about", 0, "s1.com"),
        ("https://s2.com/about", 0, "s2.com"),
        ("https://s3.com/about", 0, "s3.com"),
        ("https://t1.com/a/b", 0, "t1.com"),
        ("https://t2.com/a/b", 0, "t2.com"),
        ("https://t3.com/a/b", 0, "t3.com"),
    ]
    train_rows = [
        ("https://ptr-a.com/", 1, "ptr-a.com"),
        ("https://ptr-b.com/", 1, "ptr-b.com"),
        ("https://ptr-c.com/about", 1, "ptr-c.com"),
        ("https://ptr-d.com/a/b", 1, "ptr-d.com"),
        ("https://btr-1.com/", 0, "btr-1.com"),
        ("https://btr-2.com/", 0, "btr-2.com"),
        ("https://btr-3.com/", 0, "btr-3.com"),
        ("https://btr-4.com/", 0, "btr-4.com"),
        ("https://btr-5.com/about", 0, "btr-5.com"),
        ("https://btr-6.com/about", 0, "btr-6.com"),
        ("https://btr-7.com/a/b", 0, "btr-7.com"),
        ("https://btr-8.com/a/b", 0, "btr-8.com"),
    ]
    for rows, name in ((train_rows, "train.csv"), (test_rows, "test.csv")):
        pd.DataFrame(rows, columns=["url", "label", "registrable_domain"]).to_csv(
            base / name, index=False
        )
    return len(train_rows), len(test_rows)


def test_prematch_stats_cover_full_benign_population(tmp_path: Path) -> None:
    _write_eval_dir(tmp_path)
    res = diag.run_diagnostic(eval_dir=tmp_path, seed=0, n_boot=5)

    test = pd.read_csv(tmp_path / "test.csv")
    benign_urls = test[test.label == 0]["url"]
    assert len(benign_urls) == 12

    pre = res["netloc_len"]["pre_match_benign"]
    matched = res["netloc_len"]["matched_benign"]

    # Pre-match covers every benign test URL ...
    assert pre["n"] == 12
    expected = build_splits.shape_features(
        pd.DataFrame({"url": list(benign_urls)})
    )[:, diag.NETLOC_LEN_INDEX]
    assert pre["mean"] == float(np.mean(expected))
    assert pre["median"] == float(np.median(expected))
    assert pre["min"] == float(np.min(expected))
    assert pre["max"] == float(np.max(expected))

    # ... while the matched entry covers only the retained subset, so the
    # two populations are provably distinct entries.
    assert matched["n"] == res["matching"]["n_benign_retained"]
    assert matched["n"] < pre["n"]
    assert res["matching"]["n_benign_discarded"] == pre["n"] - matched["n"]

    for key in (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
    ):
        assert key in pre and key in matched, key
