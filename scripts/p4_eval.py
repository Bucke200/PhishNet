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
        "# Phase 4 report — close-out (phase4-D, unpublished)",
        "",
        "The recorded sweep was not run, so only a sealed provisional exists",
        "and nothing here publishes. The phase question — whether the LLM",
        "layer beats the password baseline — is unanswered, not negative.",
        "Coverage is stated beside every number (criterion 14).",
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
    # Priced forecast (never a bill): Groq's own rate page,
    # console.groq.com/docs/models, read 2026-09-18 —
    # openai/gpt-oss-120b at $0.15 input / $0.60 output per 1M tokens.
    # Reasoning bills as output. Means below are independently rounded;
    # exact cold-cache means are in the close-out log (scripts/p4_closeout.py).
    prompt_mean = float(np.mean(pt)) if pt else 0.0
    completion_mean = float(np.mean(ct)) if ct else 0.0
    reasoning_mean = float(np.mean(rt)) if rt else 0.0
    per_call_usd = prompt_mean / 1e6 * 0.15 + completion_mean / 1e6 * 0.60
    escalation_rate = 1106 / 24819
    report["cost"] = {
        "n_cold_calls": len(usages),
        "prompt_tokens_mean": prompt_mean,
        "completion_tokens_mean": completion_mean,
        "reasoning_tokens_mean": reasoning_mean,
        "visible_tokens_mean": (
            float(np.mean([c - r for c, r in zip(ct, rt, strict=True)])) if ct else 0
        ),
        "latency_p50_ms": float(np.median(latencies)) if latencies else 0,
        "latency_p90_ms": float(np.percentile(latencies, 90)) if latencies else 0,
        "rate_source": "console.groq.com/docs/models, read 2026-09-18",
        "rate_input_per_1m": 0.15,
        "rate_output_per_1m": 0.60,
        "forecast_usd_per_call": per_call_usd,
        "forecast_usd_per_1000_escalated": per_call_usd * 1000,
        "forecast_usd_per_1000_all_rows": per_call_usd * 1000 * escalation_rate,
        "forecast_usd_3x1106_repeats": per_call_usd * 3318,
        "pricing": "provisional forecast from measured counts, never a bill",
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
    cost = report["cost"]
    lines += [
        "",
        "## Lead findings (close-out, phase4-D)",
        "- Fetchability is a label proxy: test phish fetch 0.135 vs test "
        "benign 0.886 (Step-0 marginals). The takedown filter already "
        "selected for live phish; the fetch then selects again.",
        "- Structural ceiling (sealed Step-0 data, no LLM call): fetched "
        "in-band test phish 132/3799 = 0.0347 is the most recall the layer "
        "could ever add; FPR exposure is 969/21020 = 0.0461 benign. "
        "This explains 'indistinguishable' before a reader asks.",
        "- system_fingerprint rotates per call (35 values over 101 calls); "
        "run identity is model+prompt+seed 0, explicitly weaker (phase4-C).",
        "- Determinism 11/50 (22%, over the 5% bar): 10 free-tier quota "
        "failures plus 1 genuine phishing->suspicious wobble. The response "
        "cache, not the seed and not the temperature, is what makes the "
        "published numbers reproducible.",
        "- Password baseline, exact: 2 fires in 24,819 test rows (both "
        "benign; 0 of 3,799 phish), identical among the 1,158 fetched-ok "
        "rows. Not threshold degeneracy (scores are 0/1 against "
        "t_alert=0.9269, so every 1.0 fires) — the rule itself almost "
        "never fires on this population. A cascade-slot variant would be a "
        "new predictor after LLM reads and stays future work.",
        "",
        "## Descriptive (provisional)",
        f"- agreement on sealed rows: {agree:.3f} (n={len(sealed)})",
        f"- verdict distribution: {dict(dist)}",
        "- rank metrics: artifactual per §2 (tie block at t_alert); "
        "fixed-threshold recall/FPR above is primary.",
        f"- cost (cold-cache, n={len(usages)}): prompt "
        f"{cost['prompt_tokens_mean']:.2f} / completion "
        f"{cost['completion_tokens_mean']:.2f} "
        f"(reasoning {cost['reasoning_tokens_mean']:.2f}, visible "
        f"{cost['visible_tokens_mean']:.2f}) tokens per call "
        "(independently rounded means; exact: 1475.82 / 243.22 = "
        "89.58 + 153.64); "
        f"latency p50 {cost['latency_p50_ms']:.0f}ms p90 "
        f"{cost['latency_p90_ms']:.0f}ms; provisional forecast at Groq "
        "listed rates ($0.15/$0.60 per 1M, 2026-09-18): "
        f"${cost['forecast_usd_per_call']:.5f}/call, "
        f"${cost['forecast_usd_per_1000_escalated']:.3f}/1k escalated, "
        f"${cost['forecast_usd_per_1000_all_rows']:.4f}/1k rows at "
        "escalation 0.0446; 3x1106 repeats ~= "
        f"${cost['forecast_usd_3x1106_repeats']:.2f}.",
        "- coverage: verdicts sealed for "
        f"{len(verdict_by_url)}/1106 test in-band fetched-ok rows with "
        "extracts (1101 in-band by stored manifest scores plus 5 in-band "
        "step0-sample rows, stored tier1 NaN, scored identically at sweep "
        "time — verified); "
        "unjudged rows retain Tier-1 (§2 failure policy).",
        "",
        "## Criteria (close-out)",
        "- 1, 2, 4, 5, 7, 8, 9a, 11, 12: met.",
        "- 3: unmet. Achieved benign band mass 406/8110 = 0.0501; §1.2's "
        "'0.05 by construction' was wrong — floor(0.055 x 8110) = 446 and "
        "floor(0.005 x 8110) = 40 admit at most 406 rows, above 405.5, "
        "knowable at registration (phase4-D).",
        "- 6: met on provisional numbers only.",
        "- 9: met under phase4-C (weaker, stated).",
        "- 9b, 10b: unmet, per phase4-D (no recorded sweep; Developer tier "
        "unavailable, free tier ~100 calls/day).",
        "- 10: reported (22%; 10 quota failures + 1 genuine wobble).",
        "- 10a: met (unanswered, not negative; nothing beyond this model).",
        "- 13: met on provisional counts; priced forecast from Groq's own "
        "rate page, dated 2026-09-18, labeled provisional.",
        "- 14: met.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
