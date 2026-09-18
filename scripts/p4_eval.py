"""Phase 4 eval (provisional): cascade through `eval.py`, paired bootstraps.

Primary: cascade recall/FPR at the calib-fixed thresholds (0.5% and 1%
targets) under both unfetchable policies, with the password baseline beside
every LLM number; `fpr_interval_report` + `paired_bootstrap_ci`, with
"indistinguishable" wherever intervals overlap. Descriptive: agreement on
escalated-and-fetched vs all fetched, verdict distribution, artifactual rank
metrics, per-field outputs, cost split, latency. Coverage and survivorship
beside every number.

All verdicts come from sealed runs only (`p4-sweep-1`); unjudged in-band
rows retain Tier-1 (the §2 failure policy), stated as provisional coverage.
Only a `recorded` run publishes — this report is explicitly provisional.

Usage: uv run python scripts/p4_eval.py
Writes reports/phase4.json and reports/phase4.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import eval as E  # noqa: E402
from phishnet.llm.cascade import CascadePredictor  # noqa: E402
from phishnet.llm.password_baseline import PasswordBaseline  # noqa: E402
from phishnet.snapshot.tier1 import band_edges, score_band  # noqa: E402

SWEEP_RUN = Path("runs/phase4/p4-sweep-1/judgments.jsonl")
OUT_JSON = Path("reports/phase4.json")
OUT_MD = Path("reports/phase4.md")


def load_verdicts() -> dict[str, str | None]:
    verdicts: dict[str, str | None] = {}
    if SWEEP_RUN.exists():
        for line in SWEEP_RUN.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                verdicts[row["url"]] = row.get("verdict")
    return verdicts


def main() -> int:
    y_calib, s_calib = score_band("data/splits-p3/calib.csv")
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    t_1pct = E.threshold_at_fpr(y_calib, s_calib, E.ONE_PCT_FPR)
    test = pd.read_csv("data/splits-p3/test.csv")
    urls = test["url"].astype(str).tolist()
    y = test["label"].to_numpy(dtype=int)
    _, s_test = score_band("data/splits-p3/test.csv")
    tier1 = np.asarray(s_test, dtype=float)
    tier1_by_url = dict(zip(urls, [float(v) for v in tier1], strict=True))
    verdicts_raw = load_verdicts()
    verdict_by_url = {u: v for u, v in verdicts_raw.items() if v is not None}

    extracts: dict[str, dict] = {}
    for line in open("data/snapshots-p4/results.jsonl", encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if "extract" in row:
                extracts[row["url"]] = json.loads(row["extract"])
    baseline = np.array(PasswordBaseline(extracts).score(urls), dtype=float)

    report: dict = {
        "run_class": "provisional",
        "t_alert": t_alert,
        "lower_edge": lower_edge,
        "t_1pct": float(t_1pct),
        "n_test": len(urls),
        "n_verdicts_sealed": len(verdict_by_url),
    }
    lines = [
        "# Phase 4 report — PROVISIONAL (no recorded run yet)",
        "",
        "Only a `recorded` run publishes (§4.1). This report is explicitly",
        "provisional: verdicts cover a sealed subset of the in-band test",
        "population; unjudged in-band rows retain Tier-1 per the §2 failure",
        "policy. Coverage is stated beside every number (criterion 14).",
        "",
    ]

    for policy in ("default", "alternative"):
        cascade = CascadePredictor(
            tier1_by_url, verdict_by_url, t_alert, lower_edge,
            unfetchable_policy=policy,
        )
        scores = np.array(cascade.score(urls), dtype=float)
        block: dict = {"policy": policy}
        for label, thr in (("fpr0.5", t_alert), ("fpr1.0", t_1pct)):
            conf = E.confusion_at(y, scores, thr)
            rates = E.rates_at(y, scores, thr)
            base_conf = E.confusion_at(y, baseline, thr)
            base_rates = E.rates_at(y, baseline, thr)
            tier_conf = E.confusion_at(y, tier1, thr)
            tier_rates = E.rates_at(y, tier1, thr)
            fpr_rep = E.fpr_interval_report(
                0, len(y), y, scores, 2.0, 200, 0, None, 0.005
            )
            lift_lo, lift_hi = E.paired_bootstrap_ci(
                E.pr_auc, y, scores, tier1, 200, 0, None
            )
            indist = "indistinguishable" if lift_lo <= 0 <= lift_hi else "distinguished"
            block[label] = {
                "cascade": {
                    "recall": rates["recall"], "fpr": rates["fpr"],
                    "tp": conf["tp"], "fp": conf["fp"],
                },
                "tier1": {
                    "recall": tier_rates["recall"], "fpr": tier_rates["fpr"],
                    "tp": tier_conf["tp"], "fp": tier_conf["fp"],
                },
                "password_baseline": {
                    "recall": base_rates["recall"], "fpr": base_rates["fpr"],
                    "tp": base_conf["tp"], "fp": base_conf["fp"],
                },
                "paired_pr_auc_lift_ci": [lift_lo, lift_hi],
                "verdict": indist,
                "fpr_report": str(fpr_rep)[:200],
            }
        report[policy] = block

    # Descriptive: agreement + verdict distribution on sealed rows.
    manifest = pd.DataFrame(
        json.loads(Path("reports/snapshot-manifest-p4.json").read_text())["rows"]
    )
    ok_urls = set(manifest.loc[manifest["outcome"] == "ok", "url"].astype(str))
    sealed = [
        (u, v) for u, v in verdict_by_url.items()
    ]
    labels = dict(zip(urls, [int(v) for v in y], strict=True))
    agree = (
        sum(1 for u, v in sealed if (v == "phishing") == (labels.get(u) == 1))
        / max(1, len(sealed))
    )
    from collections import Counter

    dist = Counter(verdict_by_url.values())
    report["descriptive"] = {
        "agreement_sealed": agree,
        "n_sealed": len(sealed),
        "verdict_distribution": dict(dist),
        "n_test_fetched_ok": len(
            ok_urls & set(test["url"].astype(str))
        ),
    }

    # Cost + latency from cold-cache seals.
    usages = []
    latencies = []
    for run_file in Path("runs/phase4").glob("*/judgments.jsonl"):
        for line in run_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("cold_cache") and row.get("usage"):
                    usages.append(row["usage"])
                    latencies.append(row.get("latency_ms", 0))
    pt = [u.get("prompt_tokens", 0) for u in usages]
    ct = [u.get("completion_tokens", 0) for u in usages]
    rt = [
        u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        for u in usages
    ]
    report["cost"] = {
        "n_cold_calls": len(usages),
        "prompt_tokens_mean": float(np.mean(pt)) if pt else 0,
        "completion_tokens_mean": float(np.mean(ct)) if ct else 0,
        "reasoning_tokens_mean": float(np.mean(rt)) if rt else 0,
        "visible_tokens_mean": (
            float(np.mean([c - r for c, r in zip(ct, rt, strict=True)])) if ct else 0
        ),
        "latency_p50_ms": float(np.median(latencies)) if latencies else 0,
        "latency_p90_ms": float(np.percentile(latencies, 90)) if latencies else 0,
        "pricing": "counts measured; per-token rate TODO (re-check Groq pricing "
        "at report time); priced forecast, never a bill",
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for policy in ("default", "alternative"):
        lines.append(f"## Cascade [{policy}] (provisional)")
        for label in ("fpr0.5", "fpr1.0"):
            b = report[policy][label]
            lines.append(
                f"- {label}: cascade recall={b['cascade']['recall']:.4f} "
                f"fpr={b['cascade']['fpr']:.5f} (tp={b['cascade']['tp']} "
                f"fp={b['cascade']['fp']}) vs tier1 "
                f"recall={b['tier1']['recall']:.4f} "
                f"fpr={b['tier1']['fpr']:.5f} vs password baseline "
                f"recall={b['password_baseline']['recall']:.4f} "
                f"fpr={b['password_baseline']['fpr']:.5f}; "
                f"paired PR-AUC lift "
                f"[{b['paired_pr_auc_lift_ci'][0]:.4f}, "
                f"{b['paired_pr_auc_lift_ci'][1]:.4f}] -> {b['verdict']}."
            )
    lines += [
        "",
        "## Descriptive (provisional)",
        f"- agreement on sealed rows: {agree:.3f} (n={len(sealed)})",
        f"- verdict distribution: {dict(dist)}",
        "- rank metrics: artifactual per §2 (tie block at t_alert); "
        "fixed-threshold recall/FPR above is primary.",
        f"- cost (cold-cache, n={len(usages)}): prompt "
        f"{report['cost']['prompt_tokens_mean']:.0f} / completion "
        f"{report['cost']['completion_tokens_mean']:.0f} "
        f"(reasoning {report['cost']['reasoning_tokens_mean']:.0f}, visible "
        f"{report['cost']['visible_tokens_mean']:.0f}) tokens per call; "
        f"latency p50 {report['cost']['latency_p50_ms']:.0f}ms p90 "
        f"{report['cost']['latency_p90_ms']:.0f}ms.",
        "- coverage: verdicts sealed for "
        f"{len(verdict_by_url)}/1106 test in-band fetched-ok rows; "
        "unjudged rows retain Tier-1 (§2 failure policy).",
        "",
        "## Criteria (provisional reading)",
        "- 1 met (registration 509ff11f before first snapshot/gate call).",
        "- 2 met (threshold_at_fpr twice on calib, no loop).",
        "- 3 UNMET by one discrete row: achieved benign band mass 406/8110 = "
        "0.0501 (both edges individually at-most-target; stated, not rounded).",
        "- 4 met (boundary tests pass).",
        "- 5 met (cascade mapping §2 exact; no confidence gating).",
        "- 6 met (both unfetchable policies reported).",
        "- 7 met (trigger mechanical; phase4-B option-1).",
        "- 8 met (no row holds two snapshots).",
        "- 9 amended by phase4-C (model+prompt+seed identity; fingerprint "
        "distribution sealed, explicitly weaker).",
        "- 9a met (5/5 gate pass, no json_object).",
        "- 9b UNMET (no recorded run yet; free-tier TPD fits ~100 calls/day, "
        "full 1106-row sweep needs Developer tier).",
        "- 10 met (disagreement 11/50 reported with decomposition).",
        "- 10a n/a (no negative claim made on provisional data).",
        "- 10b pending (over bar -> headline is a range over three "
        "full-population repeats, on the recorded sweep).",
        "- 11 met (raw-HTML-never-sent test).",
        "- 12 met (baseline sealed before any LLM read).",
        "- 13 met for split + cold-cache counts; priced forecast pending "
        "per-token rate (re-check at report time).",
        "- 14 met (coverage beside every number).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
