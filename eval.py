"""PhishNet evaluation harness.

One command, one fixed report, any predictor.

    python eval.py --predictor predictors:LegacyEnsemble --dataset data/splits/test.csv

The report is the contract for every later phase. Nothing in here knows anything
about how a predictor works; it only calls `.score(urls) -> floats`.

Dependencies: numpy, pandas, scikit-learn. Deliberately no tldextract/requests
here — registrable domain and suffix are columns produced by build_splits.py, so
the harness stays runnable on a frozen CSV with no network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCHEMA_VERSION = "1.0.0"

REQUIRED_COLUMNS = ["url", "label", "first_seen", "registrable_domain"]


# --------------------------------------------------------------------------
# Predictor interface
# --------------------------------------------------------------------------


@runtime_checkable
class Predictor(Protocol):
    """Anything the harness can evaluate.

    `score` returns one float per URL, higher = more likely phishing. If the
    floats are probabilities in [0, 1] the report includes calibration; if they
    are unbounded scores calibration is skipped and only ranking metrics and the
    chosen operating point are reported.
    """

    name: str

    def score(self, urls: Sequence[str]) -> Sequence[float]: ...


def load_predictor(spec: str, kwargs: dict[str, str] | None = None) -> Predictor:
    """Load `module:attr`. If attr is a class, instantiate it with kwargs."""
    if ":" not in spec:
        raise ValueError(f"predictor spec must be 'module:attr', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    obj = getattr(importlib.import_module(module_name), attr)
    if isinstance(obj, type):
        obj = obj(**(kwargs or {}))
    if not hasattr(obj, "score"):
        raise TypeError(f"{spec} does not implement .score(urls)")
    if not hasattr(obj, "name"):
        obj.name = spec  # type: ignore[attr-defined]
    return obj  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class EvalConfig:
    target_fpr: float = 0.005
    # Prevalence of phishing in real browsing traffic. The test set is roughly
    # balanced; deployment is not. Precision is reported at BOTH.
    deployment_prevalence: float = 1e-4
    bootstrap: int = 1000
    seed: int = 0
    calibration_bins: int = 10
    min_slice_n: int = 50
    latency_sample: int = 300
    latency_warmup: int = 20
    batch_size: int = 512


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose achieved FPR (score >= t) stays within budget.

    Ties matter here. Interpolating the ROC curve would invent an operating
    point the scores cannot actually produce, so we walk real score values.
    """
    neg = np.sort(s[y == 0])[::-1]  # descending
    if neg.size == 0:
        raise ValueError("no negatives in evaluation set")
    budget = int(np.floor(target_fpr * neg.size))
    if budget == 0:
        return float(np.nextafter(neg[0], np.inf))
    thr = float(neg[budget - 1])
    while (neg >= thr).sum() > budget:
        higher = neg[neg > thr]
        if higher.size == 0:
            return float(np.nextafter(neg[0], np.inf))
        thr = float(higher[-1])  # smallest negative score strictly above thr
    return thr


def confusion_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, int]:
    pred = s >= thr
    return {
        "tp": int((pred & (y == 1)).sum()),
        "fp": int((pred & (y == 0)).sum()),
        "fn": int((~pred & (y == 1)).sum()),
        "tn": int((~pred & (y == 0)).sum()),
    }


def rates_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    c = confusion_at(y, s, thr)
    pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    return {
        "recall": c["tp"] / pos if pos else float("nan"),
        "fpr": c["fp"] / neg if neg else float("nan"),
        "precision": c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan"),
    }


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Precision the user would actually experience at a given base rate."""
    num = prevalence * recall
    den = num + (1 - prevalence) * fpr
    return num / den if den > 0 else float("nan")


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    # average_precision_score, not trapezoid-under-the-PR-curve: the latter
    # interpolates between operating points that do not exist.
    return float(average_precision_score(y, s))


def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def calibration(y: np.ndarray, s: np.ndarray, bins: int) -> dict[str, Any]:
    """Equal-mass (quantile) bins — equal-width bins are useless on skewed scores."""
    n = len(s)
    order = np.argsort(s, kind="stable")
    edges = np.array_split(order, min(bins, max(1, len(np.unique(s)))))
    table, ece, mce = [], 0.0, 0.0
    for idx in edges:
        if idx.size == 0:
            continue
        conf = float(s[idx].mean())
        freq = float(y[idx].mean())
        gap = abs(conf - freq)
        ece += (idx.size / n) * gap
        mce = max(mce, gap)
        table.append(
            {
                "n": int(idx.size),
                "score_lo": float(s[idx].min()),
                "score_hi": float(s[idx].max()),
                "mean_score": conf,
                "empirical_rate": freq,
                "gap": conf - freq,
            }
        )
    return {
        "ece": ece,
        "mce": mce,
        "brier": float(np.mean((s - y) ** 2)),
        "bins": table,
    }


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile CI.

    If `groups` is given, resample whole registrable domains rather than rows.
    Rows within a domain are not independent — a single phishing kit can put 200
    near-identical URLs in the test set, and row bootstrap will report a CI three
    times tighter than reality.
    """
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        for _ in range(n_boot):
            idx = np.concatenate(
                [
                    rng.choice(pos_idx, pos_idx.size, replace=True),
                    rng.choice(neg_idx, neg_idx.size, replace=True),
                ]
            )
            try:
                vals.append(fn(y[idx], s[idx]))
            except Exception:
                continue
    else:
        uniq = np.unique(groups)
        index_of = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n_boot):
            picked = rng.choice(uniq, uniq.size, replace=True)
            idx = np.concatenate([index_of[g] for g in picked])
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                vals.append(fn(y[idx], s[idx]))
            except Exception:
                continue
    if not vals:
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# --------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------


def length_bucket(url: str) -> str:
    n = len(url)
    for hi, name in [(30, "<30"), (60, "30-59"), (100, "60-99"), (200, "100-199")]:
        if n < hi:
            return name
    return ">=200"


def age_bucket(days: float) -> str:
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return "unknown"
    for hi, name in [(1, "<1d"), (7, "1-6d"), (30, "7-29d"), (365, "30-364d")]:
        if days < hi:
            return name
    return ">=365d"


def build_slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    slices: dict[str, pd.Series] = {}
    if "suffix" in df.columns:
        slices["tld"] = df["suffix"].fillna("unknown").astype(str)
    slices["url_length"] = df["url"].map(length_bucket)
    if "domain_age_days" in df.columns:
        slices["domain_age"] = df["domain_age_days"].map(age_bucket)
    if "source" in df.columns:
        slices["source"] = df["source"].fillna("unknown").astype(str)
    return slices


def slice_report(
    df: pd.DataFrame, y: np.ndarray, s: np.ndarray, thr: float, cfg: EvalConfig
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for slice_name, values in build_slices(df).items():
        rows = []
        for group, mask in values.groupby(values).groups.items():
            m = values.index.isin(mask)
            n = int(m.sum())
            yg, sg = y[m], s[m]
            row: dict[str, Any] = {
                "group": str(group),
                "n": n,
                "positives": int((yg == 1).sum()),
                "insufficient": n < cfg.min_slice_n,
            }
            row.update(rates_at(yg, sg, thr))
            row["pr_auc"] = pr_auc(yg, sg) if len(np.unique(yg)) > 1 else float("nan")
            rows.append(row)
        rows.sort(key=lambda r: -r["n"])
        out[slice_name] = rows[:25]
    return out


# --------------------------------------------------------------------------
# Scoring + latency
# --------------------------------------------------------------------------


def score_all(pred: Predictor, urls: list[str], batch_size: int) -> tuple[np.ndarray, float]:
    scores: list[float] = []
    t0 = time.perf_counter()
    for i in range(0, len(urls), batch_size):
        scores.extend(pred.score(urls[i : i + batch_size]))
    wall = time.perf_counter() - t0
    arr = np.asarray(scores, dtype=float)
    if arr.shape[0] != len(urls):
        raise ValueError(f"predictor returned {arr.shape[0]} scores for {len(urls)} urls")
    if not np.isfinite(arr).all():
        raise ValueError("predictor returned non-finite scores")
    return arr, wall


def latency_probe(pred: Predictor, urls: list[str], cfg: EvalConfig) -> dict[str, float]:
    """Single-URL call latency. A browser extension blocks on one URL, not a batch."""
    rng = np.random.default_rng(cfg.seed)
    sample = list(rng.choice(urls, min(cfg.latency_sample, len(urls)), replace=False))
    for u in sample[: cfg.latency_warmup]:
        pred.score([u])
    times = []
    for u in sample:
        t0 = time.perf_counter()
        pred.score([u])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


# --------------------------------------------------------------------------
# Warnings — things that silently invalidate a number
# --------------------------------------------------------------------------


def collect_warnings(df: pd.DataFrame, y: np.ndarray, s: np.ndarray, cfg: EvalConfig) -> list[str]:
    w: list[str] = []
    n_neg = int((y == 0).sum())
    allowed_fp = int(np.floor(cfg.target_fpr * n_neg))
    if allowed_fp < 20:
        w.append(
            f"Only {n_neg} negatives, so an FPR of {cfg.target_fpr:.3%} is {allowed_fp} false "
            f"positives. The operating point is estimated from too few events; you need "
            f"~{int(20 / cfg.target_fpr):,} negatives for a stable estimate."
        )
    uniq = int(np.unique(s).size)
    if uniq < 10:
        w.append(
            f"Predictor emits only {uniq} distinct scores. PR-AUC, ROC-AUC and the FPR sweep "
            f"are not meaningful for a step function — read the operating point only, and fix "
            f"the predictor to emit probabilities."
        )
    if s.min() < 0 or s.max() > 1:
        w.append("Scores fall outside [0,1]; calibration metrics skipped.")
    dom_overlap = None
    if "split" in df.columns:
        dom_overlap = 0
    if df["registrable_domain"].nunique() < 100:
        w.append(
            f"Test set covers only {df['registrable_domain'].nunique()} registrable domains. "
            f"Per-domain bootstrap CIs will be wide, which is honest but unstable."
        )
    top = df[df.label == 1]["registrable_domain"].value_counts()
    if len(top) and top.iloc[0] / max(1, (df.label == 1).sum()) > 0.1:
        w.append(
            f"Domain {top.index[0]!r} accounts for {top.iloc[0] / (df.label == 1).sum():.0%} of "
            f"positives — one campaign is driving the headline recall."
        )
    base = float(y.mean())
    if not 0.2 < base < 0.8:
        w.append(f"Test base rate is {base:.1%}; PR-AUC is base-rate dependent and not comparable across datasets.")
    return w


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(pred: Predictor, dataset: Path, cfg: EvalConfig) -> dict[str, Any]:
    df = pd.read_csv(dataset)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    urls = df["url"].astype(str).tolist()

    scores, wall = score_all(pred, urls, cfg.batch_size)
    lat = latency_probe(pred, urls, cfg)

    thr = threshold_at_fpr(y, scores, cfg.target_fpr)
    conf = confusion_at(y, scores, thr)
    rates = rates_at(y, scores, thr)
    groups = df["registrable_domain"].to_numpy()

    headline = {
        "pr_auc": pr_auc(y, scores),
        "roc_auc": safe_roc_auc(y, scores),
        "recall_at_target_fpr": rates["recall"],
        "achieved_fpr": rates["fpr"],
        "threshold": thr,
        "precision_on_test_set": rates["precision"],
        "precision_at_deployment_prevalence": precision_at_prevalence(
            rates["recall"], rates["fpr"], cfg.deployment_prevalence
        ),
        "false_alerts_per_10k_browsed": rates["fpr"] * 10_000 * (1 - cfg.deployment_prevalence),
    }
    ci = {
        "pr_auc": bootstrap_ci(pr_auc, y, scores, cfg.bootstrap, cfg.seed, groups),
        "recall_at_target_fpr": bootstrap_ci(
            lambda yy, ss: rates_at(yy, ss, threshold_at_fpr(yy, ss, cfg.target_fpr))["recall"],
            y,
            scores,
            cfg.bootstrap,
            cfg.seed,
            groups,
        ),
    }

    cal = None
    if 0 <= scores.min() and scores.max() <= 1:
        cal = calibration(y, scores, cfg.calibration_bins)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictor": getattr(pred, "name", pred.__class__.__name__),
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "n": len(df),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()),
            "registrable_domains": int(df["registrable_domain"].nunique()),
            "first_seen_min": str(df["first_seen"].min()),
            "first_seen_max": str(df["first_seen"].max()),
        },
        "config": asdict(cfg),
        "headline": headline,
        "ci95": {k: list(v) for k, v in ci.items()},
        "operating_point": {**conf, **rates},
        "calibration": cal,
        "slices": slice_report(df, y, scores, thr, cfg),
        "latency": lat,
        "throughput": {"urls_per_second": len(urls) / wall if wall else float("inf")},
        "warnings": collect_warnings(df, y, scores, cfg),
        "environment": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _fmt(x: Any, pct: bool = False, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "—"
        return f"{x:.2%}" if pct else f"{x:.{digits}f}"
    return str(x)


def to_markdown(rep: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    h, d, cfg = rep["headline"], rep["dataset"], rep["config"]
    ci = rep["ci95"]
    L: list[str] = []
    L.append(f"# PhishNet eval — `{rep['predictor']}`\n")
    L.append(
        f"{rep['generated_at']} · git `{rep['environment']['git_sha']}` · "
        f"dataset `{Path(d['path']).name}` sha256 `{d['sha256'][:12]}`\n"
    )
    L.append(
        f"{d['n']:,} URLs · {d['positives']:,} phishing ({d['base_rate']:.1%}) · "
        f"{d['registrable_domains']:,} registrable domains · "
        f"{d['first_seen_min'][:10]} → {d['first_seen_max'][:10]}\n"
    )

    if rep["warnings"]:
        L.append("## Read this first\n")
        for w in rep["warnings"]:
            L.append(f"- {w}")
        L.append("")

    L.append("## Headline\n")
    L.append("| Metric | Value | 95% CI (domain bootstrap) | vs baseline |")
    L.append("|---|---|---|---|")

    def delta(key: str, pct: bool) -> str:
        if not baseline:
            return "—"
        old = baseline["headline"].get(key)
        if old is None or not isinstance(old, float) or not np.isfinite(old):
            return "—"
        dv = h[key] - old
        sign = "+" if dv >= 0 else ""
        return f"{sign}{dv:.2%}" if pct else f"{sign}{dv:.4f}"

    rows = [
        ("PR-AUC (headline)", "pr_auc", False),
        ("ROC-AUC", "roc_auc", False),
        (f"Recall @ FPR≤{cfg['target_fpr']:.2%}", "recall_at_target_fpr", True),
        ("Achieved FPR", "achieved_fpr", True),
        ("Precision (test set)", "precision_on_test_set", True),
        (
            f"Precision @ prevalence {cfg['deployment_prevalence']:.4%}",
            "precision_at_deployment_prevalence",
            True,
        ),
    ]
    for label, key, pct in rows:
        c = ci.get(key)
        cis = f"[{_fmt(c[0], pct)}, {_fmt(c[1], pct)}]" if c else "—"
        L.append(f"| {label} | {_fmt(h[key], pct)} | {cis} | {delta(key, pct)} |")
    L.append("")
    L.append(
        f"Operating threshold **{h['threshold']:.6f}**. At a deployment prevalence of "
        f"{cfg['deployment_prevalence']:.4%}, this fires **{h['false_alerts_per_10k_browsed']:.1f} "
        f"false warnings per 10,000 URLs browsed**.\n"
    )

    op = rep["operating_point"]
    L.append(f"TP {op['tp']:,} · FP {op['fp']:,} · FN {op['fn']:,} · TN {op['tn']:,}\n")

    if rep["calibration"]:
        c = rep["calibration"]
        L.append("## Calibration\n")
        L.append(f"Brier {c['brier']:.4f} · ECE {c['ece']:.4f} · MCE {c['mce']:.4f}\n")
        L.append("| bin | n | mean score | empirical rate | gap |")
        L.append("|---|---|---|---|---|")
        for i, b in enumerate(c["bins"]):
            L.append(
                f"| {i} [{b['score_lo']:.3f}–{b['score_hi']:.3f}] | {b['n']:,} | "
                f"{b['mean_score']:.3f} | {b['empirical_rate']:.3f} | {b['gap']:+.3f} |"
            )
        L.append("")

    L.append("## Slices (at the global threshold)\n")
    for name, rows_ in rep["slices"].items():
        L.append(f"### {name}\n")
        L.append("| group | n | pos | recall | FPR | PR-AUC |")
        L.append("|---|---|---|---|---|---|")
        for r in rows_:
            mark = " ⚠︎" if r["insufficient"] else ""
            L.append(
                f"| {r['group']}{mark} | {r['n']:,} | {r['positives']:,} | "
                f"{_fmt(r['recall'], True)} | {_fmt(r['fpr'], True)} | {_fmt(r['pr_auc'])} |"
            )
        L.append("")
    L.append(f"⚠︎ = fewer than {cfg['min_slice_n']} rows; treat as anecdote.\n")

    lat = rep["latency"]
    L.append("## Latency (single-URL calls)\n")
    L.append(f"p50 {lat['p50_ms']:.1f} ms · p90 {lat['p90_ms']:.1f} ms · p99 {lat['p99_ms']:.1f} ms · max {lat['max_ms']:.1f} ms")
    L.append(f"\nBatch throughput: {rep['throughput']['urls_per_second']:,.0f} URLs/s\n")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PhishNet evaluation harness")
    p.add_argument("--predictor", required=True, help="module:attr, e.g. predictors:LegacyEnsemble")
    p.add_argument("--predictor-arg", action="append", default=[], metavar="K=V")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--tag", default=None, help="report filename stem (default: predictor name)")
    p.add_argument("--compare", type=Path, default=None, help="baseline report.json for deltas")
    p.add_argument("--target-fpr", type=float, default=0.005)
    p.add_argument("--prevalence", type=float, default=1e-4)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fail-under-recall",
        type=float,
        default=None,
        help="exit 1 if recall@target FPR drops below this (for CI)",
    )
    a = p.parse_args(argv)

    cfg = EvalConfig(
        target_fpr=a.target_fpr,
        deployment_prevalence=a.prevalence,
        bootstrap=a.bootstrap,
        seed=a.seed,
    )
    kwargs = dict(kv.split("=", 1) for kv in a.predictor_arg)
    pred = load_predictor(a.predictor, kwargs)

    rep = evaluate(pred, a.dataset, cfg)
    baseline = json.loads(a.compare.read_text()) if a.compare and a.compare.exists() else None

    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.tag or rep["predictor"].replace(":", "_").replace("/", "_")
    (a.out / f"{stem}.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    md = to_markdown(rep, baseline)
    (a.out / f"{stem}.md").write_text(md)
    print(md)
    print(f"\nwrote {a.out / f'{stem}.json'} and {a.out / f'{stem}.md'}", file=sys.stderr)

    if a.fail_under_recall is not None:
        r = rep["headline"]["recall_at_target_fpr"]
        if not np.isfinite(r) or r < a.fail_under_recall:
            print(f"FAIL: recall {r:.4f} < {a.fail_under_recall:.4f}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
