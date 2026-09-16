"""Validate the Common-Crawl benign corpus and its trial split.

Checks (exit 1 on any failure; JSON report always written):
* duplicate normalized URLs in the new corpus (must be zero)
* malformed URLs (must be zero)
* per-eTLD+1 concentration (reported; cap enforced at selection time)
* URL-type distribution vs the deduplicated phishing feeds (tolerance)
* scheme composition: |benign_https_rate - phishing_https_rate| must not
  exceed SCHEME_RATE_GAP_MAX (hard gate; has_port gap reported, ungated)
* mechanism hard gates: path-depth single-feature ROC-AUC two-sided
  (|AUC - 0.5| <= PATH_DEPTH_AUC_MAXDIST) and URL-length inversion
  (mean benign length minus mean phishing length >= URL_LEN_INVERSION_MIN)
* shape advisory: total shape-only ROC-AUC on the trial split (when
  --split-dir points at built train/test CSVs) reported against
  SHAPE_AUC_ADVISORY — investigate above it, never fail
* hostname/netloc stats (netloc_len, subdomain_count, hyphen/digit
  density) for the new corpus side by side with phishing
* train/test eTLD+1 overlap of the trial split (must be zero)
* stratified shape gate (Amendment D, D0.2; --mode stratified): the same
  four metrics computed main-benign vs non-hosted phishing (promotion
  blocking) and hosted-benign vs hosted phishing (descriptive only).
  The unstratified block is always computed and reported.

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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import build_splits
from build_cc_benign import url_type
from phishnet.enrichment.key import (  # type: ignore[import-untyped]
    host_of,
    is_hosted_tenant,
)

TYPE_TOLERANCE = 0.03  # max abs share drift per URL type vs phishing
SCHEME_RATE_GAP_MAX = 0.04  # max |benign_https_rate - phishing_https_rate|
# Two-sided shape parity: |single-feature ROC-AUC - 0.5| for path depth.
# Deliberately two-sided, not "AUC <= 0.55": the frozen 0.753-era splits
# carry path_depth AUC ~0.30 (benign DEEPER than phishing — an inverted
# signal a one-sided gate scores as a pass). The new CC corpus measures
# 0.4916 (|d| = 0.008). A recurrence in either direction fails.
PATH_DEPTH_AUC_MAXDIST = 0.05
# Directional length guard: mean benign URL length must be at least the
# phishing mean. This guards the bare-domain-benign failure mode; it is
# retained even though the frozen splits also pass it (+3 to +5 chars) —
# it did not produce the 0.753 era and cannot by itself catch its return.
URL_LEN_INVERSION_MIN = 0.0
# Total shape AUC is advisory only (warn, never fail): the 0.64/0.66
# residual was localized to phishing netloc structure (tail-only audit
# 0.6610 > full 0.6400 — head removal moved it the wrong way), i.e. no
# benign sampling moves it. See docs/cc-benign-acquisition.md.
SHAPE_AUC_ADVISORY = 0.70


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


def path_depths(urls: list[str]) -> list[int]:
    """Path depth per URL (same definition as build_splits.enrich)."""
    return [len([s for s in urlparse(u).path.split("/") if s]) for u in urls]


def single_feature_auc(
    benign_vals: Sequence[float], phish_vals: Sequence[float]
) -> float:
    """ROC-AUC of one continuous shape feature, benign=0 vs phishing=1."""
    y = [0] * len(benign_vals) + [1] * len(phish_vals)
    return float(roc_auc_score(y, list(benign_vals) + list(phish_vals)))


def stratum_metrics(
    benign_urls: list[str], phish_urls: list[str], blocking: bool
) -> dict[str, Any]:
    """D0.2 shape metrics for one hosted stratum (Amendment D).

    Same four quantities and tolerances as the unstratified gate, under
    the registered D0.2 names: type_drift (<= TYPE_TOLERANCE per type),
    scheme_gap (<= SCHEME_RATE_GAP_MAX), path_depth_auc_dist
    (<= PATH_DEPTH_AUC_MAXDIST), url_len_inversion (>=
    URL_LEN_INVERSION_MIN). Inputs are NORMALISED URLs on both sides
    (the phishing reference is the deduped normalised set; benign rows
    are normalised the same way), so a stratified run reproduces an
    M1-style measurement exactly. When blocking is False (hosted
    stratum: descriptive per Amendment C follow-up) metrics are reported
    but never registered as failures; empty sides report None for the
    AUC/inversion rather than raising.
    """
    types = ("root", "path1", "pathN", "query")
    n_b = len(benign_urls)
    n_p = len(phish_urls)
    if n_b:
        b_share = {
            t: sum(1 for u in benign_urls if url_type(u) == t) / n_b for t in types
        }
    else:
        b_share = {t: 0.0 for t in types}
    if n_p:
        p_share = {
            t: sum(1 for u in phish_urls if url_type(u) == t) / n_p for t in types
        }
    else:
        p_share = {t: 0.0 for t in types}
    drift = {t: abs(b_share[t] - p_share[t]) for t in types}
    gap, b_https, p_https = binary_rate_gap(benign_urls, phish_urls, is_https)
    auc: float | None = None
    dist: float | None = None
    inversion: float | None = None
    if n_b and n_p:
        auc_v = single_feature_auc(path_depths(benign_urls), path_depths(phish_urls))
        be_len = sum(len(u) for u in benign_urls) / n_b
        ph_len = sum(len(u) for u in phish_urls) / n_p
        auc, dist, inversion = auc_v, abs(auc_v - 0.5), be_len - ph_len
    failures: list[str] = []
    if blocking:
        for t, d in drift.items():
            if d > TYPE_TOLERANCE:
                failures.append(f"type_drift[{t}]={d:.4f}")
        if gap > SCHEME_RATE_GAP_MAX:
            failures.append(f"scheme_gap={gap:.4f}")
        if dist is not None and dist > PATH_DEPTH_AUC_MAXDIST:
            failures.append(f"path_depth_auc_dist={dist:.4f}")
        if inversion is not None and inversion < URL_LEN_INVERSION_MIN:
            failures.append(f"url_len_inversion={inversion:.2f}")
    return {
        "n_benign": n_b,
        "n_phish": n_p,
        "type_share": b_share,
        "phish_type_share": p_share,
        "type_drift": drift,
        "type_tolerance": TYPE_TOLERANCE,
        "scheme_gap": gap,
        "scheme_gap_max": SCHEME_RATE_GAP_MAX,
        "benign_https": b_https,
        "phish_https": p_https,
        "path_depth_auc": auc,
        "path_depth_auc_dist": dist,
        "path_depth_auc_maxdist": PATH_DEPTH_AUC_MAXDIST,
        "url_len_inversion": inversion,
        "url_len_inversion_min": URL_LEN_INVERSION_MIN,
        "blocking": blocking,
        "failures": failures,
    }


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
    p.add_argument(
        "--mode",
        choices=("unstratified", "stratified"),
        default="unstratified",
        help="stratified also gates main-benign vs non-hosted phishing "
        "(blocking, D0.2) and reports hosted vs hosted (descriptive); "
        "the unstratified block is always computed and reported",
    )
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
    # Mechanism hard gates (two-number gate spec, part 1). Measured on the
    # same populations as the scheme gate: normalized new-corpus URLs vs
    # the deduplicated phishing feeds.
    be_norms = [n for n in norms if n is not None]
    depth_auc = single_feature_auc(path_depths(be_norms), path_depths(phish_list))
    depth_dist = abs(depth_auc - 0.5)
    depth_passed = bool(depth_dist <= PATH_DEPTH_AUC_MAXDIST)
    if not depth_passed:
        failures.append(f"path_depth_auc_dist={depth_dist:.4f}")
    be_len = sum(len(n) for n in be_norms) / len(be_norms) if be_norms else 0.0
    ph_len = sum(len(n) for n in phish_list) / len(phish_list) if phish_list else 0.0
    len_inversion = be_len - ph_len
    len_passed = bool(len_inversion >= URL_LEN_INVERSION_MIN)
    if not len_passed:
        failures.append(f"url_len_inversion={len_inversion:.2f}")
    mechanism_info = {
        "path_depth_auc": depth_auc,
        "path_depth_auc_dist": depth_dist,
        "path_depth_auc_maxdist": PATH_DEPTH_AUC_MAXDIST,
        "path_depth_gate_passed": depth_passed,
        "url_len_inversion": len_inversion,
        "url_len_inversion_min": URL_LEN_INVERSION_MIN,
        "url_len_gate_passed": len_passed,
    }

    overlap: list[str] = []
    split_info: dict = {}
    shape_advisory: dict = {}
    stratified_info: dict[str, Any] | None = None
    if a.mode == "stratified":
        # D0.2, computed on NORMALISED URLs both sides (phish_list is the
        # deduped normalised reference; be_norms the normalised corpus),
        # so a run reproduces an M1-style measurement exactly.
        be_all = [n for n in norms if n is not None]
        b_main = [n for n in be_all if not is_hosted_tenant(host_of(n))]
        b_host = [n for n in be_all if is_hosted_tenant(host_of(n))]
        p_main = [u for u in phish_list if not is_hosted_tenant(host_of(u))]
        p_host = [u for u in phish_list if is_hosted_tenant(host_of(u))]
        main_m = stratum_metrics(b_main, p_main, blocking=True)
        host_m = stratum_metrics(b_host, p_host, blocking=False)
        for f in main_m["failures"]:
            failures.append(f"main:{f}")
        stratified_info = {
            "main": main_m,
            "hosted": host_m,
            "note": "main stratum promotion-blocking (D0.2); hosted stratum "
            "descriptive (Amendment C follow-up) — reported, never fails; "
            "the unstratified block above is retained for every candidate",
        }
        print(f"stratified main failures: {main_m['failures']}")
        print(
            f"main n_benign={main_m['n_benign']} n_phish={main_m['n_phish']} "
            f"hosted n_benign={host_m['n_benign']} n_phish={host_m['n_phish']}"
        )
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
        # Advisory band (two-number gate spec, part 2): total shape AUC is
        # reported with an investigate threshold, never a failure. Trial
        # CSVs do not carry path_depth, so recompute it (same definition).
        for frame in (tr, te):
            frame["path_depth"] = path_depths([str(u) for u in frame["url"].tolist()])
        audit = build_splits.leakage_audit(tr, te)
        tripped = bool(audit["shape_only_roc_auc"] > SHAPE_AUC_ADVISORY)
        shape_advisory = {
            "shape_only_roc_auc": audit["shape_only_roc_auc"],
            "shape_only_pr_auc": audit["shape_only_pr_auc"],
            "threshold": SHAPE_AUC_ADVISORY,
            "tripped": tripped,
            "audit_verdict": audit["verdict"],
            "note": "advisory only: investigate above threshold, never fail; "
            "the 0.64/0.66 residual was localized to phishing netloc "
            "structure, not benign sampling",
        }
        if tripped:
            print(
                f"ADVISORY: trial shape_only_roc_auc={audit['shape_only_roc_auc']:.4f} "
                f"exceeds {SHAPE_AUC_ADVISORY} — investigate, not a failure",
            )
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
        "mechanism_gates": mechanism_info,
        "shape_advisory": shape_advisory,
        "root_host_forms": hostform_info,
        "synthesized_roots": synth_info,
        "hostname_stats_benign": hostname_stats(urls),
        "hostname_stats_phish": hostname_stats(sorted(seen)),
        "split": split_info,
        "split_etld1_overlap_sample": overlap[:10],
        "failures": failures,
    }
    if a.out is not None:
        if stratified_info is not None:
            # Stratified-only key: unstratified reports stay byte-identical
            # to before --mode existed.
            report["stratified"] = stratified_info
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
    print(
        f"mechanisms: path_depth_auc={mechanism_info['path_depth_auc']:.4f} "
        f"(dist={mechanism_info['path_depth_auc_dist']:.4f}) "
        f"url_len_inversion={mechanism_info['url_len_inversion']:+.2f}"
    )
    if failures:
        print(f"FAILURES: {failures}", file=sys.stderr)
        return 1
    print("OK: all validation checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
