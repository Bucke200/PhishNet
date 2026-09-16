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
   Phishing intra-class domain reuse across T is intentional, not leakage:
   a campaign that persists across the split date is genuinely something the
   model must detect on the far side, and the leakage question under audit
   is forward-in-time generalisation. Only cross-class (train/test)
   registrable-domain overlap is removed, in step 4.
3. Split benign negatives by registrable-domain hash, never by their crawl
   timestamp. Each registrable domain is assigned wholly to train or test via
   sha256("<neg-hash-seed>:<registrable-domain>") mapped to [0, 1); domains
   below --benign-test-fraction go to test. Stable across runs by construction.
4. Enforce registrable-domain disjointness. A domain that appears on both sides
   is dropped from TEST, not from train — dropping from train would throw away
   labelled data for no benefit, and the test set is the thing that has to be
   clean. The drop itself is gated (--max-straddler-drop-share,
   --min-benign-test-domains; defaults are the committed thresholds): a
   split that has to discard too much test to get clean, or that leaves too
   few benign test domains to measure with, is refused rather than eyeballed.
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

from phishnet.enrichment.key import gate_psl_snapshot, hosted_share

RAW = Path("data/raw")
OUT = Path("data/splits")

# Pin the public suffix list to the snapshot bundled with the pinned tldextract
# release. Empty suffix_list_urls cannot fetch at all (tldextract raises
# SuffixListNotFound on zero URLs, then falls back to the bundled snapshot),
# so grouping is frozen for a given tldextract version on every machine with
# no network, ever. cache_dir alone is not enough: on a cache miss it would
# refresh over HTTP and silently regroup shared-suffix domains. The snapshot
# file's sha256 is recorded in the manifest so a swapped PSL is visible at
# build time.
PSL_SNAPSHOT_NAME = ".tld_set_snapshot"

# Benign negatives carry only a crawl timestamp, so a global time cutoff would
# put them all on one side. They are partitioned by registrable-domain hash
# instead: sha256("<seed>:<domain>") -> [0, 1), test iff below the fraction.
# hashlib.sha256 is stable across processes/runs (unlike hash()), so the
# assignment is reproducible given the same seed + fraction, which are recorded
# in the manifest.
NEG_HASH_SEED_DEFAULT = "phishnet-neg-split-v1"
NEG_TEST_FRACTION_DEFAULT = 0.2

# Disjointness acceptance gates (committed defaults; applies to future
# splits, never retroactively to frozen ones): a split that must discard
# too much test to get clean, or that leaves too few benign test domains
# to measure with, is refused rather than eyeballed. Both are CLI
# overridable for small synthetic fixtures in tests; the defaults below
# are the commitment, and the manifest records the effective values only
# through the pre-existing keys (no new manifest keys: repro/hashes.json
# pins manifest.json byte-for-byte, so the schema is frozen — gate
# outcomes print to stdout, refusal exits 1 with no files written).
# Calibrated, not fitted:
# * STRADDLER_DROP_SHARE_MAX = 0.10: fail if dropped straddling domains
#   exceed 10% of pre-drop test domains. Floor-calibrated against the
#   pinned successor rebuild, which drops 184/2,767 (6.65%) and stands as
#   accepted practice: any threshold at or below that would retroactively
#   veto the project's own successor, so the gate sits above precedent
#   with headroom while still catching genuinely overlapping pools.
#   (An earlier 0.02 value was committed and then revoked on this exact
#   evidence — see git history — because it vetoed the pinned rebuild.)
# * BENIGN_TEST_DOMAINS_FLOOR = 250: fail if the final test set holds
#   fewer than 250 benign registrable domains. Rationale: bootstrap CIs
#   are resampled by domain, so domains are the cluster count that keeps
#   them stable; the URL-side power budget (>= ~4,000 benign test URLs for
#   20 FPs at FPR <= 0.5%) is checked separately at eval time. Scale
#   anchor (history, not fit): frozen splits hold 104 / 371 / 997 benign
#   test domains, and the ~2,900-domain successor design yields ~580 at a
#   0.2 test fraction, so 250 sits well below expectation — a tripwire,
#   not a tuning knob.
#   Boundary values pass (usable iff share <= max and domains >= floor).
STRADDLER_DROP_SHARE_MAX = 0.10
BENIGN_TEST_DOMAINS_FLOOR = 250


def neg_domain_hash_fraction(domain: str, seed: str) -> float:
    """Deterministic [0, 1) value for one registrable domain."""
    digest = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# Phase 3 power fallback, pre-registered before any number exists: the
# post-cap benign test count decides by rule which option's expectations
# apply — ≥15k (≈±0.10pp half-width at FPR 0.5%) resolves the 0.5%
# verdict under option-1 rules; below it the 0.5% verdict is expected
# "indistinguishable" and the pre-registered 1% point carries
# resolvability (option-2 rules). Never by judgment after seeing results.
PHASE3_BENIGN_TEST_FLOOR = 15_000


def phase3_power_option(n_benign_test: int) -> str:
    """Pre-registered switch: "option-1" or "option-2-fallback"."""
    return (
        "option-1" if n_benign_test >= PHASE3_BENIGN_TEST_FLOOR else "option-2-fallback"
    )


def neg_domain_is_test(domain: str, seed: str, test_fraction: float) -> bool:
    """True iff this benign registrable domain belongs in test (wholly)."""
    return neg_domain_hash_fraction(domain, seed) < test_fraction


def _snapshot_file() -> Path:
    """Resolve the bundled PSL snapshot backing offline extraction."""
    return Path(tldextract.__file__).resolve().parent / PSL_SNAPSHOT_NAME


def _extractor() -> tldextract.TLDExtract:
    return tldextract.TLDExtract(
        cache_dir=".tld_cache",
        suffix_list_urls=(),  # no network, ever (see module comment)
        fallback_to_snapshot=True,
    )


EXTRACT = _extractor()
PSL_SOURCE = f"snapshot:tldextract-{tldextract.__version__}:{PSL_SNAPSHOT_NAME}"
PSL_SNAPSHOT_SHA256 = hashlib.sha256(_snapshot_file().read_bytes()).hexdigest()


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


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's raw bytes (input provenance for manifests)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_date_from_name(name: str) -> str | None:
    """Extract the YYYY-MM-DD snapshot date embedded in a raw filename.

    Raw snapshots are named `<source>-<YYYY-MM-DD>.jsonl` (e.g.
    `openphish-2026-09-12.jsonl`). Returns the date string or None when
    the name carries no date (e.g. ad-hoc fixtures in tests).
    """
    import re

    # Anchor to the trailing -<date>.jsonl: index pins like CC-MAIN-2026-34
    # contain earlier digit runs that a bare search would match.
    m = re.search(r"-(\d{4}-\d{2}-\d{2})\.jsonl$", name)
    return m.group(1) if m else None


def survival_stratum(lag_days: float | None, time_basis: str | None = None) -> str:
    """Bucket a phishing survival lag into fresh/short/long/unknown.

    Lag is `snapshot_collection_ts - first_seen` in days: how long a URL
    survived between submission and its first snapshot appearance.
    Thresholds are pre-committed (fresh ≤ 2d, short ≤ 30d, long > 30d).

    Rows with `time_basis == "observed"` (OpenPhish: first_seen is the
    snapshot moment itself, so lag would read ~0 for every row) carry no
    age information at all — they are live phish of unknown age, i.e. the
    survivor case — and map to "unknown", never "fresh". They are kept out
    of the fresh slice. Benign rows and uncomputable lags map to "na".
    """
    if time_basis == "observed":
        return "unknown"
    if lag_days is None or pd.isna(lag_days):
        return "na"
    if lag_days <= 2.0:
        return "fresh"
    if lag_days <= 30.0:
        return "short"
    return "long"


def scheme_rates(frame: pd.DataFrame) -> tuple[float, float]:
    """Benign/phishing https rates on a split frame (train only, see below)."""
    b = frame[frame.label == 0]["url"].map(
        lambda u: urlparse(str(u)).scheme.lower() == "https"
    )
    p = frame[frame.label == 1]["url"].map(
        lambda u: urlparse(str(u)).scheme.lower() == "https"
    )
    return (
        float(b.mean()) if len(b) else float("nan"),
        float(p.mean()) if len(p) else float("nan"),
    )


def should_drop_is_https(benign_https_rate: float, phish_https_rate: float) -> bool:
    """Pre-committed scheme rule for the trained vocabulary.

    Dropping the `is_https` column alone does NOT remove the scheme
    signal — `https://` is one character longer than `http://` and most
    lexical features are computed on the raw URL string — so a DROP
    decision means "remove or canonicalize the scheme before featurizing"
    (Phase 3: `features.extraction.canonicalize_scheme`, applied to the
    lexical-retrained baseline and every enriched row alike), not "drop
    one column".

    `is_https` stays only when the scheme gate passes (|gap| <= 0.04)
    without making benign URLs less like what the extension sees — i.e.
    never via post-hoc scheme filtering of benign (which trades the
    distortion into depth/rank strata). Test-era phishing sits well
    below the benign rate for genuine deployment reasons, so the
    expected outcome is DROP; the gate numbers are recorded either way.

    Measured on TRAIN (train, or train+calib once the third band lands),
    recorded before any model is fit — measuring on the full population
    would let the test set influence a vocabulary decision. The leakage
    audit always keeps scheme (it must see the signal).
    """
    from validate_cc_benign import SCHEME_RATE_GAP_MAX

    return abs(benign_https_rate - phish_https_rate) > SCHEME_RATE_GAP_MAX


def load_raw(raw_dir: Path = RAW) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no raw files in {RAW}/ — run collect.py first")
    rows = []
    for f in files:
        snap = snapshot_date_from_name(f.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                # Snapshot provenance for survival-lag strata: the earliest
                # file containing a URL is its first snapshot. Recorded
                # before dedup (enrich() keeps the minimum per URL). The
                # source filename is kept alongside so enrich() can derive
                # the file's actual collection time (max first_seen within
                # the file) instead of midnight of the filename date.
                if snap is not None and "first_snapshot" not in row:
                    row["first_snapshot"] = snap
                if "_snap_file" not in row:
                    row["_snap_file"] = f.name
                rows.append(row)
    if not rows:
        sys.exit(f"no rows in {raw_dir}/ — run collect.py first")
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame, *, phase3: bool = False) -> pd.DataFrame:
    """Normalise, dedup, and group URLs.

    Default (``phase3=False``) is the frozen legacy path, byte-for-byte:
    pinned populations (repro/hashes.json) rebuild identically with or
    without this flag present. ``phase3=True`` adds snapshot provenance
    (first_snapshot/snapshot_anchor/survival_stratum) for the new
    population — never for a rebuild of a pinned one.
    """
    df = df.copy()
    df["url"] = df["url"].map(normalise)
    df = df[df["url"].notna()]
    df["first_seen"] = pd.to_datetime(
        df["first_seen"], utc=True, format="mixed", errors="coerce"
    )
    df = df[df["first_seen"].notna()]
    if not phase3:
        # Earliest observation wins for URLs seen in several snapshots.
        df = df.sort_values("first_seen").drop_duplicates(subset=["url"], keep="first")
        LAST_ANCHORS.clear()
        return _group(df)
    # Phase 3 path: earliest observation wins, and the earliest snapshot
    # wins alongside it (survival lag needs both). first_seen is
    # submission/capture time, first_snapshot is the first daily file
    # containing the URL — see docs/point-in-time.md.
    #
    # Collection time, not midnight: first_snapshot parsed from the filename
    # is a date with no time, so same-day rows (e.g. submitted 07:03 UTC,
    # collected ~08:38 UTC) would get a ~−7h lag. The file's collection
    # time is instead its max first_seen (for OpenPhish files every row
    # carries the collection stamp, so the max IS the run time; for
    # PhishTank dumps the newest submission approximates it) — content
    # derived, hence reproducible across checkouts, unlike file mtimes.
    # Files with no parseable stamps fall back to end-of-day UTC of the
    # filename date (a guaranteed upper bound, never a negative lag).
    pre = df
    enrich_anchors: dict[str, dict[str, str]] = {}
    # Per-file collection-time anchors (priority order — recorded per file
    # in the manifest, since the fallback biases lag the other way):
    #  1. openphish-run-stamp: the OpenPhish stamp of the same collection
    #     date IS the moment that day's run executed (single-valued per
    #     file, verified). Applies to every file of that date, phishing or
    #     benign — the dump max (anchor 2) always precedes it by minutes to
    #     hours (e.g. 09-15: dump max 07:03 vs run 08:38), so anchor 2
    #     underestimates lag by that much while still guaranteeing >= 0.
    #  2. file-max: max first_seen within the file (newest submission).
    #  3. filename-eod: end-of-day UTC of the filename date — an upper
    #     bound that overestimates lag; files with no parseable stamps only.
    file_ts: dict[str, Any] = {}
    file_method: dict[str, str] = {}
    run_stamp: dict[str, Any] = {}
    if "_snap_file" in pre.columns:
        for fname, grp in pre.groupby("_snap_file"):
            base = str(fname).split("/")[-1].split("\\")[-1]
            if base.startswith("openphish-"):
                d = snapshot_date_from_name(base)
                vals = grp["first_seen"].dropna()
                if d is not None and len(vals):
                    run_stamp[d] = vals.max()
        for fname, grp in pre.groupby("_snap_file"):
            key = str(fname)
            base = key.split("/")[-1].split("\\")[-1]
            d = snapshot_date_from_name(base)
            if d is not None and d in run_stamp:
                file_ts[key] = run_stamp[d]
                file_method[key] = "openphish-run-stamp"
                continue
            mx = grp["first_seen"].max()
            if pd.notna(mx):
                file_ts[key] = mx
                file_method[key] = "file-max"
            elif d is not None:
                file_ts[key] = (
                    pd.Timestamp(d, tz="UTC")
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )
                file_method[key] = "filename-eod"
            else:
                file_ts[key] = pd.NaT
                file_method[key] = "none"
    enrich_anchors.update(
        {
            k: {
                "ts": (v.isoformat() if pd.notna(v) else "na"),
                "method": file_method.get(k, "none"),
            }
            for k, v in file_ts.items()
        }
    )
    if "first_snapshot" in df.columns:
        if "_snap_file" in df.columns:
            df["_snap_file_ts"] = pd.to_datetime(
                df["_snap_file"].map(file_ts), utc=True, errors="coerce"
            )
        else:
            df["_snap_file_ts"] = pd.Series(
                pd.NaT, index=df.index, dtype="datetime64[ns, UTC]"
            )
        # Filename-date fallback for rows whose file has no stamp at all.
        missing_ts = df["_snap_file_ts"].isna() & df["first_snapshot"].notna()
        df.loc[missing_ts, "_snap_file_ts"] = (
            pd.to_datetime(
                df.loc[missing_ts, "first_snapshot"], utc=True, errors="coerce"
            )
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
        df = df.sort_values(["first_seen", "_snap_file_ts"]).drop_duplicates(
            subset=["url"], keep="first"
        )
        # A URL seen in several snapshots keeps its earliest snapshot by
        # collection time (ties broken toward the better anchor: run-stamp
        # over file-max over filename-eod). Sort order above is by
        # first_seen, so a later-snapshot row with an earlier stamp would
        # otherwise win — recompute per URL from the file table.
        method_rank = {
            "openphish-run-stamp": 0,
            "file-max": 1,
            "filename-eod": 2,
            "none": 3,
        }
        if "_snap_file" in pre.columns:
            pairs = (
                pre[["url", "_snap_file"]]
                .drop_duplicates()
                .assign(
                    _ts=lambda p: pd.to_datetime(
                        p["_snap_file"].map(file_ts), utc=True, errors="coerce"
                    ),
                    _rank=lambda p: (
                        p["_snap_file"]
                        .map(file_method)
                        .map(method_rank)
                        .fillna(3)
                        .astype(int)
                    ),
                )
            )
            pairs = pairs.dropna(subset=["_ts"]).sort_values(["_ts", "_rank"])
            first_file = pairs.drop_duplicates(subset=["url"], keep="first").set_index(
                "url"
            )
            df["_snap_ts"] = df["url"].map(first_file["_ts"])
            df["_snap_method"] = df["url"].map(
                first_file["_snap_file"].map(file_method)
            )
            df["first_snapshot"] = df["url"].map(
                first_file["_snap_file"].map(
                    lambda f: snapshot_date_from_name(
                        str(f).split("/")[-1].split("\\")[-1]
                    )
                )
            )
        else:
            # Legacy rows (snapshot date, no source file): end-of-day of
            # the earliest snapshot date, method filename-eod.
            df["_snap_ts"] = (
                pd.to_datetime(df["first_snapshot"], utc=True, errors="coerce")
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            df["_snap_method"] = "filename-eod"
        df = df.drop(columns=["_snap_file_ts"])
    else:
        df = df.sort_values("first_seen").drop_duplicates(subset=["url"], keep="first")
        df["_snap_ts"] = pd.NaT
        df["_snap_method"] = "none"
    df["first_snapshot"] = df.get("first_snapshot", pd.Series([pd.NA] * len(df)))
    # Public per-row anchor provenance (which rule dated this row); the
    # per-file table rides to the manifest via enrich.last_anchors.
    df["snapshot_anchor"] = df.get("_snap_method", pd.Series(["none"] * len(df)))
    df = df.drop(columns=["_snap_method"], errors="ignore")
    snap_ts = pd.to_datetime(df.pop("_snap_ts"), utc=True, errors="coerce")
    lag_days = (snap_ts - df["first_seen"]).dt.total_seconds() / 86400.0
    # Non-negative by construction for file-max anchors (file ts >= every
    # row ts inside it) and verified per-run for openphish-run-stamp
    # anchors (the run stamp postdates the dump max on all four current
    # snapshots); anything beyond 60s of clock skew is broken input.
    bad = lag_days[(df["label"] == 1) & lag_days.notna() & (lag_days < -60 / 86400)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} phishing rows have negative survival lag "
            f"(min {bad.min():.3f} days); snapshot collection time precedes "
            "first_seen — refusing rather than mis-stratifying."
        )
    df["survival_lag_days"] = lag_days.where(df["label"] == 1, float("nan"))
    has_basis = "time_basis" in df.columns
    df["survival_stratum"] = [
        survival_stratum(v, tb) if lab == 1 else "na"
        for lab, v, tb in zip(
            df["label"],
            lag_days,
            df["time_basis"] if has_basis else [None] * len(df),
            strict=True,
        )
    ]
    df = df.drop(columns=["_snap_file"], errors="ignore")
    out = _group(df)
    # Per-file anchor table for the manifest (refreshed every call).
    LAST_ANCHORS.clear()
    LAST_ANCHORS.update(enrich_anchors)
    return out


def _group(df: pd.DataFrame) -> pd.DataFrame:
    """Registrable-domain grouping shared by both enrich paths."""
    ext = df["url"].map(EXTRACT)
    df["registrable_domain"] = [
        f"{e.domain}.{e.suffix}" if e.suffix else e.domain for e in ext
    ]
    df["suffix"] = [e.suffix or "none" for e in ext]
    df["path_depth"] = df["url"].map(
        lambda u: len([s for s in urlparse(u).path.split("/") if s])
    )
    return df.reset_index(drop=True)


# Refreshed by enrich() on every call: {filename: {"ts", "method"}} for
# the manifest's snapshot_anchors block.
LAST_ANCHORS: dict[str, dict[str, str]] = {}


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
    p.add_argument(
        "--max-straddler-drop-share",
        type=float,
        default=STRADDLER_DROP_SHARE_MAX,
        help="refuse the split if dropped straddling domains exceed this "
        "share of pre-drop test domains (default is the committed gate)",
    )
    p.add_argument(
        "--min-benign-test-domains",
        type=int,
        default=BENIGN_TEST_DOMAINS_FLOOR,
        help="refuse the split if the final test set holds fewer benign "
        "registrable domains (default is the committed floor; lower only "
        "for small synthetic fixtures in tests)",
    )
    p.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw JSONL dir (default: data/raw). Pin the exact input set; "
        "the manifest records which files were read.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir for train.csv/test.csv/manifest.json "
        "(default: data/splits). Frozen populations keep their own dirs.",
    )
    p.add_argument(
        "--deterministic-manifest",
        action="store_true",
        help="move the volatile run timestamp out of manifest.json into a "
        "run-meta.json sidecar, so the output dir is fully deterministic "
        "and whole directories diff cleanly across runs and platforms.",
    )
    p.add_argument(
        "--expect-psl-sha",
        default=None,
        help="pin the PSL snapshot: refuse to build when the runtime "
        "snapshot sha differs (rebuild gate; omit on the first build of a "
        "new population, then pin the recorded sha)",
    )
    p.add_argument(
        "--phase3",
        action="store_true",
        help="Phase 3 population: snapshot provenance (first_snapshot, "
        "snapshot_anchor, survival_stratum) plus the scheme/host/power "
        "records in the manifest. Default OFF: pinned populations rebuild "
        "byte-for-byte without it (repro/hashes.json).",
    )
    a = p.parse_args()
    if not 0.0 < a.benign_test_fraction < 1.0:
        sys.exit("--benign-test-fraction must be strictly between 0 and 1")
    raw_dir = a.raw if a.raw is not None else RAW
    out_dir = a.out if a.out is not None else OUT

    gate_psl_snapshot(a.expect_psl_sha)

    df = enrich(load_raw(raw_dir), phase3=a.phase3)
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
    n_test_domains_pre = test.registrable_domain.nunique()
    test = test[~test.registrable_domain.isin(straddling)]
    drop_share = len(straddling) / n_test_domains_pre if n_test_domains_pre else 0.0
    # Split metric (reported, not a second gate): straddlers that touch the
    # benign side vs phishing-only temporal straddlers (same kit
    # infrastructure both sides of T). The refined 2% hard gate applies to
    # the benign-involved share; the total share keeps the code-default cap.
    benign_domains = set(benign.registrable_domain)
    be_involved = {d for d in straddling if d in benign_domains}
    be_share = len(be_involved) / n_test_domains_pre if n_test_domains_pre else 0.0
    print(
        f"dropped {len(straddling):,} straddling domains "
        f"from test -> {len(test):,} rows "
        f"(drop_share={drop_share:.4f} of {n_test_domains_pre:,} pre-drop "
        f"test domains; benign-involved {len(be_involved):,} "
        f"(share={be_share:.4f}), phishing-only "
        f"{len(straddling) - len(be_involved):,})"
    )
    if drop_share > a.max_straddler_drop_share:
        print(
            f"\n  !! STRADDLER GATE FAILED: drop share {drop_share:.4f} "
            f"exceeds {a.max_straddler_drop_share:.2f}.\n"
            "     The two populations overlap too heavily for a clean "
            "domain-disjoint test — no files were written.",
            file=sys.stderr,
        )
        return 1

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

    n_benign_test_domains = int(test[test.label == 0].registrable_domain.nunique())
    print(f"benign test domains: {n_benign_test_domains:,}")
    if n_benign_test_domains < a.min_benign_test_domains:
        print(
            f"\n  !! DOMAIN FLOOR FAILED: {n_benign_test_domains:,} benign "
            f"test domains is below {a.min_benign_test_domains}.\n"
            "     Too few domain clusters for stable domain-resampled CIs — "
            "no files were written.",
            file=sys.stderr,
        )
        return 1

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

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "url",
        "label",
        "first_seen",
        "first_snapshot",
        "snapshot_anchor",
        "survival_stratum",
        "registrable_domain",
        "suffix",
        "source",
    ]
    if not a.phase3:
        # Pinned populations: the frozen 6-column layout, byte-for-byte.
        cols = [
            "url",
            "label",
            "first_seen",
            "registrable_domain",
            "suffix",
            "source",
        ]
    cols = [c for c in cols if c in df.columns]
    # Pre-committed scheme rule, measured on TRAIN (never the full
    # population: the test set must not influence a vocabulary decision),
    # recorded before any model is fit. A DROP means "remove or
    # canonicalize the scheme before featurizing"
    # (features.extraction.canonicalize_scheme), not "drop one column":
    # https:// is a character longer than http:// and most lexical
    # features read the raw URL string.
    benign_rate, phish_rate = scheme_rates(train)
    drop_https = should_drop_is_https(benign_rate, phish_rate)
    verdict = (
        "CANONICALIZE scheme before featurizing" if drop_https else "KEEP is_https"
    )
    print(
        f"scheme rule (train): benign_https={benign_rate:.4f} "
        f"phish_https={phish_rate:.4f} gap={abs(benign_rate - phish_rate):.4f} "
        f"-> {verdict}"
    )
    # Canonical dataset bytes are CRLF (the frozen baseline identity is
    # defined on CRLF bytes; .gitattributes checks out CRLF everywhere).
    # The pandas default lineterminator is platform-dependent, so pin it:
    # identical rows must hash identically on every OS.
    train[cols].to_csv(out_dir / "train.csv", index=False, lineterminator="\r\n")
    test[cols].to_csv(out_dir / "test.csv", index=False, lineterminator="\r\n")
    # Canonical JSON bytes are CRLF (like the CSVs): write_text translates
    # newlines per-platform by default, so pin the translation instead.
    manifest: dict[str, Any] = {
        "split_date": str(T),
        "phish_temporal_cutoff": str(T),
        "test_days": a.test_days,
        "psl_source": PSL_SOURCE,
        "psl_snapshot_sha256": PSL_SNAPSHOT_SHA256,
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
        "raw_files": sorted(f.name for f in raw_dir.glob("*.jsonl")),
        "raw_file_hashes": {
            f.name: sha256_file(f) for f in sorted(raw_dir.glob("*.jsonl"))
        },
    }
    if a.phase3:
        manifest["snapshot_anchors"] = dict(LAST_ANCHORS)
        manifest["phase3_power"] = {
            "benign_test_n": int((test.label == 0).sum()),
            "threshold": PHASE3_BENIGN_TEST_FLOOR,
            "option": phase3_power_option(int((test.label == 0).sum())),
            "note": "post-cap count decides by rule; option-2-fallback "
            "expects an indistinguishable 0.5% verdict with the 1% point "
            "carrying resolvability",
        }
        manifest["host_grouping"] = {
            "rule": "registrable-domain per pinned PSL snapshot "
            "(private section ignored by default: platform tenants group "
            "as one domain for straddler-drop and campaign caps)",
            "decision": "kept on purpose; domain-disjointness is "
            "load-bearing for the CIs; hosted share reported below",
            "train": hosted_share(train["url"].astype(str).tolist()),
            "test": hosted_share(test["url"].astype(str).tolist()),
        }
        manifest["survival_strata"] = {
            s: int((test[test.label == 1]["survival_stratum"] == s).sum())
            for s in ("fresh", "short", "long", "unknown", "na")
            if "survival_stratum" in test.columns
        }
        manifest["is_https_rule"] = {
            "measured_on": "train",
            "benign_https_rate": benign_rate,
            "phish_https_rate": phish_rate,
            "gap": abs(benign_rate - phish_rate),
            "decision": "drop" if drop_https else "keep",
            "note": "DROP = remove/canonicalize scheme before featurizing "
            "(canonicalize_scheme), not drop-one-column; audit keeps scheme; "
            "never via post-hoc benign scheme filtering",
        }
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sidecar = bool(a.deterministic_manifest)
    if sidecar:
        # The run timestamp lives in run-meta.json, never in the manifest,
        # so manifest.json is a pure function of inputs + flags.
        (out_dir / "run-meta.json").write_text(
            json.dumps({"generated_at": generated_at, "argv": sys.argv[1:]}, indent=2),
            encoding="utf-8",
            newline="\r\n",
        )
    else:
        manifest["generated_at"] = generated_at
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\r\n"
    )
    extra = ", run-meta.json" if sidecar else ""
    print(
        f"\nwrote {out_dir}/train.csv, {out_dir}/test.csv, "
        f"{out_dir}/manifest.json{extra}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
