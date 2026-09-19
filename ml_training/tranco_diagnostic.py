"""Tranco rank-tier hostname/netloc diagnostic (fully offline).

Compares hostname-shape characteristics of two long-tail Tranco tiers
against the phishing evaluation population (and, separately, the current
head-sampled benign population) to test whether head/popularity sampling
contributes to the observed shape gap.

Reads ONLY the frozen artifact
``data/raw/tranco-46VQX-top1000000-2026-09-13.csv`` plus one test CSV
(default the committed ``data/splits-eval/test.csv``; pass
``--test-csv`` for another population, e.g. the Phase 3 row-(e)
diagnostic on ``data/splits-p3/test.csv``). Makes no network requests:
parsing uses ``urlparse`` and the snapshot-pinned
``build_splits.EXTRACT`` (which is constructed with
``suffix_list_urls=()`` and can never fetch).

This is a diagnostic task only. It does NOT modify datasets, splits,
models, features, or training configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_splits  # noqa: E402

SCOPE_LIMITATION = (
    "This diagnostic evaluates hostname/netloc-shape characteristics "
    "only. It does not establish anything about `is_https`, URL path "
    "shape, query parameters, page content, or other non-hostname features."
)

REFERENCE_NOTE = (
    "Tranco is used here as a reference population, not as a labeled "
    "benign training dataset."
)

TRANC0_PATH = Path("data/raw/tranco-46VQX-top1000000-2026-09-13.csv")
TRANC0_ID = "46VQX"
TRANC0_SHA256 = "4fb2f1c0644673cf2730161ee71d835916f9da0de6fdadc8b29026160fd81d2b"

# Tier boundaries (inclusive). The requested ranges "10,000-100,000" and
# "100,000-1,000,000" overlap at rank 100,000, so the non-overlapping
# interpretation is used and documented here: A = 10,000-99,999
# (90,000 ranks), B = 100,000-1,000,000 (900,001 ranks). Head ranks
# 1-5900 are excluded from both tiers.
TIER_A = (10_000, 99_999)
TIER_B = (100_000, 1_000_000)

# subdomain_count is reported for transparency but excluded from the
# popularity-artifact verdict: Tranco list entries are bare registrable
# domains (subdomain count 0 by construction), while phishing and
# current-benign entries are full crawled URLs where subdomains naturally
# occur. That comparison has a structural limitation and is not evidence
# about popularity either way.
EVIDENCE_METRICS = ("netloc_len", "hyphen_density", "digit_density")

# Column of netloc_len in build_splits.shape_features output
# ([len(url), len(netloc), len(path), ...]); the existing implementation.
NETLOC_LEN_INDEX = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rank_map(path: Path, expected_sha256: str) -> dict[int, str]:
    """Parse the frozen list offline; refuse on sha256 mismatch."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SystemExit(f"sha256 mismatch for {path} — refusing to proceed")
    mapping: dict[int, str] = {}
    for line in content.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rank_raw, sep, domain = line.partition(",")
        if not sep:
            continue
        try:
            mapping[int(rank_raw.strip())] = domain.strip()
        except ValueError:
            continue
    return mapping


def sample_tier(
    mapping: dict[int, str],
    bounds: tuple[int, int],
    n: int,
    seed: int,
) -> list[str]:
    """Reproducible sample: default_rng(seed).choice without replacement.

    The pool is every rank in the inclusive bounds; the returned sample is
    sorted so outputs are stable regardless of sampling order.
    """
    lo, hi = bounds
    pool = [mapping[r] for r in range(lo, hi + 1) if r in mapping]
    if len(pool) != hi - lo + 1:
        raise SystemExit(f"tier {bounds}: only {len(pool)}/{hi - lo + 1} ranks present")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return sorted(pool[i] for i in idx)


def hostname_metrics(urls: list[str]) -> dict[str, np.ndarray]:
    """Hostname/netloc metrics reusing existing project implementations.

    * netloc_len: ``build_splits.shape_features`` column 1
      (``len(urlparse(url).netloc)`` — the same values the leakage audit
      and the path-depth diagnostic use).
    * subdomain_count: the exact formula from
      ``comprehensive_phishing_features`` (``subdomain.count(".") + 1``
      when a subdomain exists, else 0), evaluated with the repo's pinned
      offline extractor ``build_splits.EXTRACT`` instead of a bare
      ``tldextract.extract`` call so no network is possible.
    * hyphen/digit density: diagnostic-only ratios over the same
      ``urlparse(url).netloc`` string (no pre-existing density feature
      exists; the parsing itself is shared, not duplicated).
    """
    frame = pd.DataFrame({"url": urls})
    shape = build_splits.shape_features(frame)
    netloc_len = np.asarray(shape[:, NETLOC_LEN_INDEX], dtype=float)
    netlocs = [urlparse(u).netloc for u in urls]
    hyphen = np.asarray(
        [n.count("-") / len(n) if n else 0.0 for n in netlocs], dtype=float
    )
    digit = np.asarray(
        [sum(c.isdigit() for c in n) / len(n) if n else 0.0 for n in netlocs],
        dtype=float,
    )
    sub: list[float] = []
    for u in urls:
        host = urlparse(u).hostname or ""
        e = build_splits.EXTRACT(host)
        sub.append(float(e.subdomain.count(".") + 1) if e.subdomain else 0.0)
    return {
        "netloc_len": netloc_len,
        "subdomain_count": np.asarray(sub, dtype=float),
        "hyphen_density": hyphen,
        "digit_density": digit,
    }


def describe(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled std); sign keeps direction."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def run_diagnostic(
    tranco_path: Path = TRANC0_PATH,
    eval_test: Path = Path("data/splits-eval/test.csv"),
    n_per_tier: int = 5000,
    seed: int = 0,
    output_json: Path | None = None,
) -> dict[str, Any]:
    mapping = load_rank_map(tranco_path, TRANC0_SHA256)
    tier_a = sample_tier(mapping, TIER_A, n_per_tier, seed)
    tier_b = sample_tier(mapping, TIER_B, n_per_tier, seed)

    test = pd.read_csv(eval_test)
    phish_urls = list(test[test.label == 1]["url"])
    benign_urls = list(test[test.label == 0]["url"])

    pops = {
        "phishing": phish_urls,
        "tranco_10k_99k": [f"https://{d}/" for d in tier_a],
        "tranco_100k_1M": [f"https://{d}/" for d in tier_b],
        "current_benign": benign_urls,
    }
    metrics = {name: hostname_metrics(urls) for name, urls in pops.items()}
    stats = {
        name: {m: describe(v) for m, v in mm.items()} for name, mm in metrics.items()
    }

    # Separability of phishing from each benign-like population on one
    # hostname feature: the same max(auc, 1 - auc) convention as the
    # path-depth diagnostic's single_feature_roc_auc. AUC near 0.5 means the
    # two hostname distributions overlap heavily on that feature.
    comparison: dict[str, Any] = {}
    closer_flags: dict[str, bool] = {}
    for m in ("netloc_len", "subdomain_count", "hyphen_density", "digit_density"):
        ref = metrics["phishing"][m]
        entry: dict[str, Any] = {}
        for name in ("tranco_10k_99k", "tranco_100k_1M", "current_benign"):
            v = metrics[name][m]
            y = np.concatenate([np.ones(len(ref)), np.zeros(len(v))])
            s = np.concatenate([ref, v])
            auc = float(roc_auc_score(y, s))
            entry[name] = {
                "median_diff_vs_phishing": float(np.median(v) - np.median(ref)),
                "cohens_d_vs_phishing": cohens_d(v, ref),
                "single_feature_auc_vs_phishing": max(auc, 1.0 - auc),
            }
        for tier in ("tranco_10k_99k", "tranco_100k_1M"):
            key = f"{tier}_closer_than_current_benign"
            entry[key] = abs(entry[tier]["single_feature_auc_vs_phishing"] - 0.5) < abs(
                entry["current_benign"]["single_feature_auc_vs_phishing"] - 0.5
            )
            if m in EVIDENCE_METRICS:
                closer_flags.setdefault(tier, True)
                closer_flags[tier] = closer_flags[tier] and bool(entry[key])
            else:
                entry[key + "_structural_limitation"] = (
                    "Tranco entries are bare registrable domains; "
                    "excluded from the popularity verdict"
                )
        comparison[m] = entry

    # Verdict is computed over EVIDENCE_METRICS only (netloc_len and the
    # two densities); subdomain_count is reported but structurally limited
    # (see EVIDENCE_METRICS note) and is not popularity evidence.
    popularity_answer = (
        "Long-tail Tranco tiers are materially closer to the phishing "
        "hostname distribution than the current head-sampled benign "
        "population on every evidence metric (netloc_len, hyphen_density, "
        "digit_density); this supports the hypothesis that the existing "
        "benign collection has a popularity/head-sampling artifact. This "
        "diagnostic does not prove or disprove dataset representativeness "
        "overall."
        if all(closer_flags.values())
        else "The long-tail Tranco tiers are NOT consistently closer to "
        "phishing than the current benign population on the evidence "
        "metrics (netloc_len, hyphen_density, digit_density); this "
        "diagnostic does not support the popularity/head-sampling artifact "
        "hypothesis. It does not prove or disprove dataset "
        "representativeness overall."
    )

    results: dict[str, Any] = {
        "scope_limitation": SCOPE_LIMITATION,
        "reference_note": REFERENCE_NOTE,
        "reproducibility": {
            "tranco_input_path": str(tranco_path).replace("\\", "/"),
            "tranco_id": TRANC0_ID,
            "tranco_sha256": TRANC0_SHA256,
            "rank_boundaries": {"tier_a": list(TIER_A), "tier_b": list(TIER_B)},
            "sample_sizes": {
                "tier_a": len(tier_a),
                "tier_b": len(tier_b),
                "phishing": len(phish_urls),
                "current_benign": len(benign_urls),
            },
            "random_seed": seed,
            "sampling_method": "numpy default_rng(seed).choice(pool, "
            "size=n, replace=False); pool = every rank in inclusive "
            "bounds; output sorted",
            "feature_implementation": "netloc_len via "
            "build_splits.shape_features col 1; subdomain_count via "
            "build_splits.EXTRACT with the "
            "comprehensive_phishing_features formula; hyphen/digit "
            "density = diagnostic-only ratios over urlparse().netloc",
            "phishing_input_population": str(eval_test).replace("\\", "/")
            + " label==1",
            "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "populations": stats,
        "comparison_vs_phishing": comparison,
        "popularity_artifact_answer": popularity_answer,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, indent=2), encoding="utf-8", newline="\r\n"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tranco rank-tier hostname diagnostic (offline)"
    )
    parser.add_argument("--n-per-tier", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/splits-eval/test.csv"),
        help="test population supplying phishing + current-benign URLs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/tranco-diagnostic.json"),
    )
    args = parser.parse_args()
    res = run_diagnostic(
        eval_test=args.test_csv,
        n_per_tier=args.n_per_tier,
        seed=args.seed,
        output_json=args.out,
    )
    print(f"\n{SCOPE_LIMITATION}\n")
    print(f"{REFERENCE_NOTE}\n")
    r = res["reproducibility"]["sample_sizes"]
    print(
        f"samples: phishing={r['phishing']} "
        f"tierA(10k-99,999)={r['tier_a']} tierB(100k-1M)={r['tier_b']} "
        f"current_benign={r['current_benign']} seed={args.seed}"
    )
    for m, entry in res["comparison_vs_phishing"].items():
        row = f"  {m:<15s}"
        for name in ("tranco_10k_99k", "tranco_100k_1M", "current_benign"):
            d = entry[name]
            row += (
                f" | {name.split('_')[0][:4]}:"
                f"auc={d['single_feature_auc_vs_phishing']:.3f}"
                f" d={d['cohens_d_vs_phishing']:+.2f}"
            )
        print(row)
    print(f"\n{res['popularity_artifact_answer']}")
    if args.out:
        print(f"\nWrote diagnostic report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
