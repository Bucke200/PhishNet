"""Validate the Common-Crawl benign corpus and its trial split.

Checks (exit 1 on any failure; JSON report always written):
* duplicate normalized URLs in the new corpus (must be zero)
* malformed URLs (must be zero)
* per-eTLD+1 concentration (reported; cap enforced at selection time)
* URL-type distribution vs the deduplicated phishing feeds (tolerance)
* scheme composition: |benign_https_rate - phishing_https_rate| must not
  exceed SCHEME_RATE_GAP_MAX (hard gate; has_port gap reported, ungated)
* hostname/netloc stats (netloc_len, subdomain_count, hyphen/digit
  density) for the new corpus side by side with phishing
* train/test eTLD+1 overlap of the trial split (must be zero)

Gating convention (do not violate): binary / low-cardinality features gate
on the rate gap, continuous features gate on ROC-AUC. For a binary
feature AUC is just 0.5 + gap / 2, so an AUC-style threshold (e.g. 0.55)
compresses the whole [0, 1] gap range into [0.5, 1.0] and silently never
fires on the high side — the threshold's fireability is not inspectable
without the closed form. A gap threshold is.

Reuses build_splits.normalise / build_splits.EXTRACT /
build_splits.shape_features and build_cc_benign.url_type — no duplicated
feature logic. Never modifies data; never trains.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import build_splits
from build_cc_benign import url_type

TYPE_TOLERANCE = 0.03  # max abs share drift per URL type vs phishing
SCHEME_RATE_GAP_MAX = 0.04  # max |benign_https_rate - phishing_https_rate|


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_https(url: str) -> bool:
    return urlparse(url).scheme == "https"


def has_port(url: str) -> bool:
    return ":" in urlparse(url).netloc


def binary_rate_gap(
    benign_urls: list[str],
    phish_urls: list[str],
    indicator: Callable[[str], bool],
) -> tuple[float, float, float]:
    """Rate gap for a binary URL indicator. Returns (gap, benign_rate, phish_rate).

    This is the gated quantity for binary features (see the module-level
    gating convention): it is two-sided and its fireability is inspectable,
    unlike an AUC threshold on a 0/1 feature.
    """
    b_rate = (
        sum(1 for u in benign_urls if indicator(u)) / len(benign_urls)
        if benign_urls
        else 0.0
    )
    p_rate = (
        sum(1 for u in phish_urls if indicator(u)) / len(phish_urls)
        if phish_urls
        else 0.0
    )
    return abs(b_rate - p_rate), b_rate, p_rate


def scheme_only_auc(
    benign_urls: list[str], phish_urls: list[str]
) -> tuple[float, float, float]:
    """Single-feature (is_https) ROC-AUC, benign=0 vs phishing=1.

    Reported for comparability with the shape-only leakage audit, which
    uses the same classifier family (LogisticRegression). NOT the gated
    quantity — see binary_rate_gap and the module-level convention.

    Why any scheme gate exists: Common Crawl preferentially records https
    captures, so the per-domain observed schemes feeding the synthesised
    roots are not a neutral estimate of what a domain serves. Without a
    gate, that crawler bias becomes a class signal (benign skews https).
    """
    b = np.asarray([1.0 if urlparse(u).scheme == "https" else 0.0 for u in benign_urls])
    p = np.asarray([1.0 if urlparse(u).scheme == "https" else 0.0 for u in phish_urls])
    b_rate = float(b.mean())
    p_rate = float(p.mean())
    x = np.concatenate([b, p]).reshape(-1, 1)
    y = np.concatenate([np.zeros_like(b), np.ones_like(p)])
    scores = LogisticRegression().fit(x, y).predict_proba(x)[:, 1]
    return float(roc_auc_score(y, scores)), b_rate, p_rate


def hostname_stats(urls: list[str]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame({"url": urls})
    shape = build_splits.shape_features(frame)
    netloc_len = np.asarray(shape[:, 1], dtype=float)
    netlocs = [urlparse(u).netloc for u in urls]
    hyphen = np.asarray(
        [n.count("-") / len(n) if n else 0.0 for n in netlocs], dtype=float
    )
    digit = np.asarray(
        [sum(c.isdigit() for c in n) / len(n) if n else 0.0 for n in netlocs],
        dtype=float,
    )
    sub = []
    for u in urls:
        e = build_splits.EXTRACT(urlparse(u).hostname or "")
        sub.append(float(e.subdomain.count(".") + 1) if e.subdomain else 0.0)
    out = {}
    for name, arr in (
        ("netloc_len", netloc_len),
        ("subdomain_count", np.asarray(sub)),
        ("hyphen_density", hyphen),
        ("digit_density", digit),
    ):
        out[name] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benign", type=Path, required=True)
    p.add_argument(
        "--phish-glob", default="data/raw/openphish-*.jsonl data/raw/phishtank-*.jsonl"
    )
    p.add_argument("--split-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    failures: list[str] = []
    benign = load_jsonl(a.benign)
    urls = [r["url"] for r in benign]
    norms = [build_splits.normalise(u) for u in urls]
    malformed = sum(1 for n in norms if n is None)
    if malformed:
        failures.append(f"malformed={malformed}")
    dupes = len(norms) - len(set(norms))
    if dupes:
        failures.append(f"duplicate_urls={dupes}")

    etld1: Counter = Counter()
    for n in norms:
        if n is None:
            continue
        e = build_splits.EXTRACT(urlparse(n).hostname or "")
        etld1[f"{e.domain}.{e.suffix}".lower() if e.suffix else e.domain] += 1
    top_etld1 = etld1.most_common(5)

    types = Counter(url_type(u) for u in urls)
    n_b = len(urls)
    type_share = {t: types.get(t, 0) / n_b for t in ("root", "path1", "pathN", "query")}

    # Deduplicated phishing reference (same normalise + buckets).
    seen: set[str] = set()
    phish_types: Counter = Counter()
    for pattern in a.phish_glob.split():
        for f in sorted(glob.glob(pattern)):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    nrm = build_splits.normalise(json.loads(line)["url"])
                    if nrm is None or nrm in seen:
                        continue
                    seen.add(nrm)
                    phish_types[url_type(nrm)] += 1
    n_p = len(seen)
    phish_share = {t: phish_types.get(t, 0) / n_p for t in type_share}
    drift = {t: abs(type_share[t] - phish_share[t]) for t in type_share}
    for t, d in drift.items():
        if d > TYPE_TOLERANCE:
            failures.append(f"type_drift[{t}]={d:.4f}")

    def _scheme_rate(us: list[str]) -> float:
        return sum(1 for u in us if urlparse(u).scheme == "https") / len(us)

    def _host_forms(us: list[str]) -> dict[str, int]:
        out: Counter = Counter()
        for u in us:
            h = (urlparse(u).hostname or "").lower()
            e = build_splits.EXTRACT(h)
            reg = f"{e.domain}.{e.suffix}".lower() if e.suffix and e.domain else h
            key = "apex" if h == reg else ("www" if h == "www." + reg else "other-sub")
            out[key] += 1
        return dict(out)

    benign_roots = [u for u in urls if url_type(u) == "root"]
    phish_list = sorted(seen)
    phish_roots = [u for u in phish_list if url_type(u) == "root"]
    scheme_gap, b_https, p_https = binary_rate_gap(urls, phish_list, is_https)
    scheme_auc, _, _ = scheme_only_auc(urls, phish_list)
    scheme_passed = bool(scheme_gap <= SCHEME_RATE_GAP_MAX)
    if not scheme_passed:
        failures.append(f"scheme_rate_gap={scheme_gap:.4f}")
    port_gap, b_port, p_port = binary_rate_gap(urls, phish_list, has_port)
    scheme_info = {
        "benign_is_https_overall": b_https,
        "phish_is_https_overall": p_https,
        "benign_is_https_roots": _scheme_rate(benign_roots) if benign_roots else None,
        "phish_is_https_roots": _scheme_rate(phish_roots) if phish_roots else None,
        "scheme_rate_gap": scheme_gap,
        "scheme_rate_gap_max": SCHEME_RATE_GAP_MAX,
        "scheme_gate_passed": scheme_passed,
        "scheme_only_roc_auc": scheme_auc,
        "has_port_gap": port_gap,
        "benign_has_port_rate": b_port,
        "phish_has_port_rate": p_port,
    }
    hostform_info = {
        "benign_root_host_forms": _host_forms(benign_roots),
        "phish_root_host_forms": _host_forms(phish_roots),
    }
    synth_info = {
        "n_synthesized_roots": sum(1 for r in benign if r.get("synthesized_root")),
        "synthesized_share": sum(1 for r in benign if r.get("synthesized_root")) / n_b,
    }

    overlap: list[str] = []
    split_info: dict = {}
    if a.split_dir is not None and (a.split_dir / "test.csv").exists():
        tr = pd.read_csv(a.split_dir / "train.csv")
        te = pd.read_csv(a.split_dir / "test.csv")
        overlap = sorted(set(tr["registrable_domain"]) & set(te["registrable_domain"]))
        if overlap:
            failures.append(f"split_etld1_overlap={len(overlap)}")
        split_info = {
            "train": len(tr),
            "test": len(te),
            "train_phish": int((tr.label == 1).sum()),
            "test_phish": int((te.label == 1).sum()),
            "overlap": len(overlap),
        }
    elif a.split_dir is not None:
        split_info = {"note": "trial split not built (gate may have refused)"}

    report = {
        "benign_corpus": str(a.benign),
        "n_rows": n_b,
        "all_label_0": bool(all(r.get("label") == 0 for r in benign)),
        "malformed": malformed,
        "duplicate_urls": dupes,
        "unique_etld1": len(etld1),
        "max_urls_per_etld1": max(etld1.values()) if etld1 else 0,
        "top_etld1": [[d, c] for d, c in top_etld1],
        "url_type_counts": {t: types.get(t, 0) for t in type_share},
        "url_type_share": type_share,
        "phish_url_type_share": phish_share,
        "phish_n_deduped": n_p,
        "type_drift": drift,
        "type_tolerance": TYPE_TOLERANCE,
        "scheme_rates": scheme_info,
        "root_host_forms": hostform_info,
        "synthesized_roots": synth_info,
        "hostname_stats_benign": hostname_stats(urls),
        "hostname_stats_phish": hostname_stats(sorted(seen)),
        "split": split_info,
        "split_etld1_overlap_sample": overlap[:10],
        "failures": failures,
    }
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\r\n")
        print(f"wrote {a.out}")
    print(
        f"rows={n_b} malformed={malformed} dupes={dupes} "
        f"etld1={len(etld1)} max_per_etld1={report['max_urls_per_etld1']}"
    )
    print(f"type_share={type_share}")
    print(f"phish_share={phish_share} drift={drift}")
    print(f"split={split_info} overlap={len(overlap)}")
    if failures:
        print(f"FAILURES: {failures}", file=sys.stderr)
        return 1
    print("OK: all validation checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
