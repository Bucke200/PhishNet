"""Turn the raw append-only log into train/test splits that don't lie.

    python build_splits.py --test-days 21

Four things happen here, in order, and each one is reported as a row count so
the shrinkage is visible:

1. Normalise and deduplicate URLs.
2. Split by time: train is everything before T, test is everything on or after T.
3. Enforce registrable-domain disjointness. A domain that appears on both sides
   is dropped from TEST, not from train — dropping from train would throw away
   labelled data for no benefit, and the test set is the thing that has to be
   clean.
4. Cap URLs per domain in the test set. One phishing kit routinely emits
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
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def _extractor() -> tuple[tldextract.TLDExtract, str]:
    live = tldextract.TLDExtract(suffix_list_urls=(PSL_URL,), cache_dir=".tld_cache")
    try:
        live("example.co.uk")
        return live, PSL_URL
    except Exception:
        print(f"PSL fetch failed; using the snapshot bundled with tldextract "
              f"{tldextract.__version__}", file=sys.stderr)
        return tldextract.TLDExtract(suffix_list_urls=()), f"bundled:tldextract-{tldextract.__version__}"


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
    df["first_seen"] = pd.to_datetime(df["first_seen"], utc=True, format="mixed", errors="coerce")
    df = df[df["first_seen"].notna()]
    # Earliest observation wins for URLs seen in several snapshots.
    df = df.sort_values("first_seen").drop_duplicates(subset=["url"], keep="first")
    ext = df["url"].map(EXTRACT)
    df["registrable_domain"] = [f"{e.domain}.{e.suffix}" if e.suffix else e.domain for e in ext]
    df["suffix"] = [e.suffix or "none" for e in ext]
    df["path_depth"] = df["url"].map(lambda u: len([s for s in urlparse(u).path.split("/") if s]))
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


def leakage_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    Xtr, Xte = shape_features(train), shape_features(test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), train.label)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    out = {
        "shape_only_pr_auc": float(average_precision_score(test.label, s)),
        "shape_only_roc_auc": float(roc_auc_score(test.label, s)),
        "base_rate": float(test.label.mean()),
        "mean_path_depth_benign": float(test[test.label == 0].path_depth.mean()),
        "mean_path_depth_phish": float(test[test.label == 1].path_depth.mean()),
        "mean_url_len_benign": float(test[test.label == 0].url.str.len().mean()),
        "mean_url_len_phish": float(test[test.label == 1].url.str.len().mean()),
    }
    out["verdict"] = (
        "LEAKING" if out["shape_only_roc_auc"] > 0.85 else
        "suspicious" if out["shape_only_roc_auc"] > 0.75 else "ok"
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split-date", default=None, help="ISO date T; test is >= T")
    p.add_argument("--test-days", type=int, default=21, help="used if --split-date is absent")
    p.add_argument("--max-urls-per-domain-test", type=int, default=5)
    p.add_argument("--max-urls-per-domain-train", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = enrich(load_raw())
    print(f"loaded {len(df):,} unique URLs "
          f"({int((df.label == 1).sum()):,} phish / {int((df.label == 0).sum()):,} benign)")

    T = (
        pd.Timestamp(a.split_date, tz="UTC")
        if a.split_date
        else pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=a.test_days))
    )
    train, test = df[df.first_seen < T].copy(), df[df.first_seen >= T].copy()
    print(f"temporal split at {T.date()}: train {len(train):,} / test {len(test):,}")

    straddling = set(train.registrable_domain) & set(test.registrable_domain)
    test = test[~test.registrable_domain.isin(straddling)]
    print(f"dropped {len(straddling):,} straddling domains from test -> {len(test):,} rows")

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
        print(f"{name}: {len(frame):,} rows, {frame.label.mean():.1%} phish, "
              f"{frame.registrable_domain.nunique():,} domains")

    audit = leakage_audit(train, test)
    print("\nleakage audit (URL shape only, no phishing knowledge):")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if audit["verdict"] == "LEAKING":
        print(
            "\n  !! LEAKING: a model that only sees length and path depth is separating\n"
            "     your classes, so the split is NOT valid and no files were written.\n"
            "     Crawl more deep links per benign domain, or subsample benign to match\n"
            "     the phishing path-depth histogram, and rebuild before training or\n"
            "     evaluating anything.",
            file=sys.stderr,
        )
        return 1
    if audit["verdict"] != "ok":
        print(
          "\n  !! A model that only sees length and path depth is separating your classes.\n"
          "     Crawl more deep links per benign domain, or subsample benign to match the\n"
          "     phishing path-depth histogram, and rebuild before training anything.",
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
                "psl_source": PSL_SOURCE,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_train": len(train),
                "n_test": len(test),
                "straddling_domains_dropped": len(straddling),
                "caps": {"test": a.max_urls_per_domain_test, "train": a.max_urls_per_domain_train},
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
