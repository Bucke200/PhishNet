"""Turn the raw append-only log into train/test splits that don't lie.

    python build_splits.py --test-days 21

The negative class is time-invariant by construction: benign URLs are
collected contemporaneously, so temporal splitting is applied to phishing
positives while benign negatives are deterministically partitioned by
registrable-domain hash.

Five things happen here, in order, and each one is reported as a row count so
the shrinkage is visible:

1. Normalise and deduplicate URLs.
2. Split phishing positives by time: phish train is positives before T, phish
   test is positives on or after T. T comes from --split-date, else now minus
   --test-days. The cutoff is computed from the phishing timestamp field only.
3. Split benign negatives by registrable-domain hash, never by their crawl
   timestamp. Each registrable domain is assigned wholly to train or test via
   sha256("<neg-hash-seed>:<registrable-domain>") mapped to [0, 1); domains
   below --benign-test-fraction go to test. Stable across runs by construction.
4. Enforce registrable-domain disjointness. A domain that appears on both sides
   is dropped from TEST, not from train — dropping from train would throw away
   labelled data for no benefit, and the test set is the thing that has to be
   clean.
5. Cap URLs per domain in the test set. One phishing kit routinely emits
   hundreds of URLs under one domain; uncapped, a single campaign decides your
   headline recall and the number swings 20 points week to week.

Then it runs a leakage audit: a logistic regression on URL *shape only* — length,
path depth, query count, port, scheme — trained on train and scored on test. That
model knows nothing about phishing. If it scores well, your two classes were
collected differently and every downstream number is measuring the collection
process rather than the phenomenon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RAW = Path("data/raw")
OUT = Path("data/splits")

# Pin the public suffix list. tldextract's default behaviour is to fetch the live
# PSL, so registrable-domain grouping silently changes between runs and an old
# split stops being reproducible. Fetch once into .tld_cache/ (commit it), and
# record which source was used in the manifest.
PSL_URL = "https://publicsuffix.org/list/public_suffix_list.dat"

# Benign negatives carry only a crawl timestamp, so a global time cutoff would
# put them all on one side. They are partitioned by registrable-domain hash
# instead: sha256("<seed>:<domain>") -> [0, 1), test iff below the fraction.
# hashlib.sha256 is stable across processes/runs (unlike hash()), so the
# assignment is reproducible given the same seed + fraction, which are recorded
# in the manifest.
NEG_HASH_SEED_DEFAULT = "phishnet-neg-split-v1"
NEG_TEST_FRACTION_DEFAULT = 0.2


def neg_domain_hash_fraction(domain: str, seed: str) -> float:
    """Deterministic [0, 1) value for one registrable domain."""
    digest = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def neg_domain_is_test(domain: str, seed: str, test_fraction: float) -> bool:
    """True iff this benign registrable domain belongs in test (wholly)."""
    return neg_domain_hash_fraction(domain, seed) < test_fraction


def _extractor() -> tuple[tldextract.TLDExtract, str]:
    live = tldextract.TLDExtract(suffix_list_urls=(PSL_URL,), cache_dir=".tld_cache")
    try:
        live("example.co.uk")
        return live, PSL_URL
    except Exception:
        print(
            f"PSL fetch failed; using the snapshot bundled with tldextract "
            f"{tldextract.__version__}",
            file=sys.stderr,
        )
        return tldextract.TLDExtract(
            suffix_list_urls=()
        ), f"bundled:tldextract-{tldextract.__version__}"


EXTRACT, PSL_SOURCE = _extractor()


def normalise(url: str) -> str | None:
    try:
        p = urlparse(url.strip())
    except Exception:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    netloc = p.netloc.lower()
    if netloc.endswith(":80") and p.scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and p.scheme == "https":
        netloc = netloc[:-4]
    # Drop the fragment (never sent to a server), keep path/query case.
    return urlunparse((p.scheme, netloc, p.path or "/", p.params, p.query, ""))


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("*.jsonl"))
    if not files:
        sys.exit(f"no raw files in {RAW}/ — run collect.py first")
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["url"] = df["url"].map(normalise)
    df = df[df["url"].notna()]
    df["first_seen"] = pd.to_datetime(
        df["first_seen"], utc=True, format="mixed", errors="coerce"
    )
    df = df[df["first_seen"].notna()]
    # Earliest observation wins for URLs seen in several snapshots.
    df = df.sort_values("first_seen").drop_duplicates(subset=["url"], keep="first")
    ext = df["url"].map(EXTRACT)
    df["registrable_domain"] = [
        f"{e.domain}.{e.suffix}" if e.suffix else e.domain for e in ext
    ]
    df["suffix"] = [e.suffix or "none" for e in ext]
    df["path_depth"] = df["url"].map(
        lambda u: len([s for s in urlparse(u).path.split("/") if s])
    )
    return df.reset_index(drop=True)


def shape_features(df: pd.DataFrame) -> np.ndarray:
    def row(u: str) -> list[float]:
        p = urlparse(u)
        return [
            len(u),
            len(p.netloc),
            len(p.path),
            len([s for s in p.path.split("/") if s]),
            len(p.query),
            p.query.count("=") if p.query else 0,
            1.0 if p.scheme == "https" else 0.0,
            1.0 if ":" in p.netloc else 0.0,
        ]

    return np.asarray([row(u) for u in df["url"]], dtype=float)


def leakage_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    Xtr, Xte = shape_features(train), shape_features(test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), train.label)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    out: dict[str, Any] = {
        "shape_only_pr_auc": float(average_precision_score(test.label, s)),
        "shape_only_roc_auc": float(roc_auc_score(test.label, s)),
        "base_rate": float(test.label.mean()),
        "mean_path_depth_benign": float(test[test.label == 0].path_depth.mean()),
        "mean_path_depth_phish": float(test[test.label == 1].path_depth.mean()),
        "mean_url_len_benign": float(test[test.label == 0].url.str.len().mean()),
        "mean_url_len_phish": float(test[test.label == 1].url.str.len().mean()),
    }
    out["verdict"] = (
        "LEAKING"
        if out["shape_only_roc_auc"] > 0.85
        else "suspicious"
        if out["shape_only_roc_auc"] > 0.75
        else "ok"
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split-date",
        default=None,
        help="ISO date T; phishing positives with first_seen >= T go to test",
    )
    p.add_argument(
        "--test-days",
        type=int,
        default=21,
        help="used if --split-date is absent: T is now minus this many days",
    )
    p.add_argument(
        "--benign-test-fraction",
        type=float,
        default=NEG_TEST_FRACTION_DEFAULT,
        help="fraction of benign registrable domains assigned to test [0, 1]",
    )
    p.add_argument(
        "--neg-hash-seed",
        default=NEG_HASH_SEED_DEFAULT,
        help="seed mixed into the benign domain-hash partition",
    )
    p.add_argument("--max-urls-per-domain-test", type=int, default=5)
    p.add_argument("--max-urls-per-domain-train", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not 0.0 < a.benign_test_fraction < 1.0:
        sys.exit("--benign-test-fraction must be strictly between 0 and 1")

    df = enrich(load_raw())
    print(
        f"loaded {len(df):,} unique URLs "
        f"({int((df.label == 1).sum()):,} phish / "
        f"{int((df.label == 0).sum()):,} benign)"
    )

    T = (
        pd.Timestamp(a.split_date, tz="UTC")
        if a.split_date
        else pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=a.test_days))
    )
    phish = df[df.label == 1].copy()
    benign = df[df.label == 0].copy()
    phish_train = phish[phish.first_seen < T].copy()
    phish_test = phish[phish.first_seen >= T].copy()
    print(
        f"phishing temporal split at {T.date()}: "
        f"train {len(phish_train):,} / test {len(phish_test):,}"
    )

    # Benign crawl timestamps are ~all "now": splitting on them would strand
    # every negative on one side. Partition whole registrable domains by
    # stable hash instead; no per-URL randomness, no rebalancing.
    test_domains = {
        d
        for d in benign.registrable_domain.unique()
        if neg_domain_is_test(d, a.neg_hash_seed, a.benign_test_fraction)
    }
    benign_test = benign[benign.registrable_domain.isin(test_domains)].copy()
    benign_train = benign[~benign.registrable_domain.isin(test_domains)].copy()
    print(
        f"benign domain-hash split "
        f"(seed={a.neg_hash_seed!r}, "
        f"test_fraction={a.benign_test_fraction}): "
        f"train {len(benign_train):,} / test {len(benign_test):,} "
        f"across {benign.registrable_domain.nunique():,} domains"
    )

    train = pd.concat([phish_train, benign_train]).reset_index(drop=True)
    test = pd.concat([phish_test, benign_test]).reset_index(drop=True)
    print(f"combined: train {len(train):,} / test {len(test):,}")

    straddling = set(train.registrable_domain) & set(test.registrable_domain)
    test = test[~test.registrable_domain.isin(straddling)]
    print(
        f"dropped {len(straddling):,} straddling domains "
        f"from test -> {len(test):,} rows"
    )

    def cap(frame: pd.DataFrame, k: int) -> pd.DataFrame:
        return (
            frame.sample(frac=1.0, random_state=a.seed)
            .groupby("registrable_domain", sort=False)
            .head(k)
            .reset_index(drop=True)
        )

    before = len(test)
    test = cap(test, a.max_urls_per_domain_test)
    train = cap(train, a.max_urls_per_domain_train)
    print(f"campaign cap: test {before:,} -> {len(test):,}, train -> {len(train):,}")

    for name, frame in (("train", train), ("test", test)):
        if frame.label.nunique() < 2:
            sys.exit(f"{name} split has a single class — widen the window")
        print(
            f"{name}: {len(frame):,} rows, {frame.label.mean():.1%} phish, "
            f"{frame.registrable_domain.nunique():,} domains"
        )

    audit = leakage_audit(train, test)
    print("\nleakage audit (URL shape only, no phishing knowledge):")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if audit["verdict"] == "LEAKING":
        print(
            "\n  !! LEAKING: a model that only sees length and path depth\n"
            "     is separating your classes — the split is NOT valid and\n"
            "     no files were written.\n"
            "     Crawl more deep links per benign domain, or subsample\n"
            "     benign to match the phishing path-depth histogram, and\n"
            "     rebuild before training or evaluating anything.",
            file=sys.stderr,
        )
        return 1
    if audit["verdict"] != "ok":
        print(
            "\n  !! A model that only sees length and path depth is\n"
            "     separating your classes.\n"
            "     Crawl more deep links per benign domain, or subsample\n"
            "     benign to match the phishing path-depth histogram, and\n"
            "     rebuild before training anything.",
            file=sys.stderr,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["url", "label", "first_seen", "registrable_domain", "suffix", "source"]
    cols = [c for c in cols if c in df.columns]
    train[cols].to_csv(OUT / "train.csv", index=False)
    test[cols].to_csv(OUT / "test.csv", index=False)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "split_date": str(T),
                "phish_temporal_cutoff": str(T),
                "test_days": a.test_days,
                "psl_source": PSL_SOURCE,
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "n_train": len(train),
                "n_test": len(test),
                "n_train_phish": int((train.label == 1).sum()),
                "n_train_benign": int((train.label == 0).sum()),
                "n_test_phish": int((test.label == 1).sum()),
                "n_test_benign": int((test.label == 0).sum()),
                "benign_split": {
                    "method": "registrable-domain-hash",
                    "rule": (
                        "sha256('<seed>:<registrable_domain>') first 8 "
                        "bytes / 2**64 < test_fraction -> test; "
                        "whole domain assigned together"
                    ),
                    "seed": a.neg_hash_seed,
                    "test_fraction": a.benign_test_fraction,
                },
                "seed": a.seed,
                "straddling_domains_dropped": len(straddling),
                "caps": {
                    "test": a.max_urls_per_domain_test,
                    "train": a.max_urls_per_domain_train,
                },
                "leakage_audit": audit,
                "raw_files": sorted(f.name for f in RAW.glob("*.jsonl")),
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}/train.csv, {OUT}/test.csv, {OUT}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
