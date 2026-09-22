"""Phase 4 eval: cascade through `eval.py`, paired bootstraps.

Primary: cascade recall/FPR at the calib-fixed thresholds (0.5% and 1%
targets) under both unfetchable policies, with the password baseline beside
every LLM number; `fpr_interval_report` + `paired_bootstrap_ci`, with
"indistinguishable" wherever intervals overlap.

Repeats: a `recorded` run holds three full-population repeats (criterion
10b, because determinism measured 22% > 5%). Every primary number is
therefore reported as a **range over repeats** with per-repeat detail; a
single-repeat run (the old `p4-sweep-1`) degenerates to a point.

Serving anchor: `reports/phase4.json` is read by the serving container for
its thresholds. A run that must not overwrite it writes elsewhere via
`--out-json/--out-md`; the default paths keep the registered behavior.

Usage:
  uv run python scripts/p4_eval.py                       # p4-sweep-1
  uv run python scripts/p4_eval.py --run-id p4-recorded \
      --out-json reports/phase4-recorded.json --out-md reports/phase4-recorded.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import eval as E  # noqa: E402
from phishnet.llm.cascade import CascadePredictor  # noqa: E402
from phishnet.llm.password_baseline import PasswordBaseline  # noqa: E402
from phishnet.snapshot.tier1 import band_edges, score_band  # noqa: E402

RUNS = Path("runs/phase4")
DEFAULT_RUN_ID = "p4-sweep-1"
LABELS = (("fpr0.5", "t_alert"), ("fpr1.0", "t_1pct"))


def load_verdicts_by_repeat(path: Path) -> dict[int, dict[str, str | None]]:
    """`repeat_idx -> url -> verdict`; missing index reads as repeat 0."""
    out: dict[int, dict[str, str | None]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                idx = int(row.get("repeat_idx", 0))
                out.setdefault(idx, {})[str(row["url"])] = row.get("verdict")
    return out


def _minmax(values: list[float]) -> list[float]:
    return [float(min(values)), float(max(values))]


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _block_for_repeat(
    verdicts: dict[str, str | None],
    urls: list[str],
    y: np.ndarray,
    tier1: np.ndarray,
    baseline: np.ndarray,
    tier1_by_url: dict[str, float],
    t_alert: float,
    lower_edge: float,
    t_1pct: float,
    policy: str,
) -> dict[str, dict[str, object]]:
    verdict_by_url = {u: v for u, v in verdicts.items() if v is not None}
    cascade = CascadePredictor(
        tier1_by_url, verdict_by_url, t_alert, lower_edge, unfetchable_policy=policy
    )
    scores = np.array(cascade.score(urls), dtype=float)
    out: dict[str, dict[str, object]] = {}
    for label, thr_name in LABELS:
        thr = t_alert if thr_name == "t_alert" else t_1pct
        conf = E.confusion_at(y, scores, thr)
        rates = E.rates_at(y, scores, thr)
        base_conf = E.confusion_at(y, baseline, thr)
        base_rates = E.rates_at(y, baseline, thr)
        tier_conf = E.confusion_at(y, tier1, thr)
        tier_rates = E.rates_at(y, tier1, thr)
        fpr_rep = E.fpr_interval_report(0, len(y), y, scores, 2.0, 200, 0, None, 0.005)
        lift_lo, lift_hi = E.paired_bootstrap_ci(
            E.pr_auc, y, scores, tier1, 200, 0, None
        )
        indist = lift_lo <= 0 <= lift_hi
        out[label] = {
            "cascade": {
                "recall": rates["recall"],
                "fpr": rates["fpr"],
                "tp": conf["tp"],
                "fp": conf["fp"],
            },
            "tier1": {
                "recall": tier_rates["recall"],
                "fpr": tier_rates["fpr"],
                "tp": tier_conf["tp"],
                "fp": tier_conf["fp"],
            },
            "password_baseline": {
                "recall": base_rates["recall"],
                "fpr": base_rates["fpr"],
                "tp": base_conf["tp"],
                "fp": base_conf["fp"],
            },
            "paired_pr_auc_lift_ci": [lift_lo, lift_hi],
            "indistinguishable": bool(indist),
            "fpr_report": str(fpr_rep)[:200],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-json", default="reports/phase4.json")
    parser.add_argument("--out-md", default="reports/phase4.md")
    args = parser.parse_args(argv)

    run_dir = RUNS / args.run_id
    run_meta: dict = {}
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    verdicts_by_repeat = load_verdicts_by_repeat(run_dir / "judgments.jsonl")
    if not verdicts_by_repeat:
        verdicts_by_repeat = {0: {}}
    repeat_indexes = sorted(verdicts_by_repeat)
    n_repeats = int(run_meta.get("n_repeats", len(repeat_indexes)))
    run_class = str(run_meta.get("run_class", "provisional"))
    coverage = {
        str(r): sum(1 for v in verdicts_by_repeat[r].values() if v is not None)
        for r in repeat_indexes
    }

    y_calib, s_calib = score_band("data/splits-p3/calib.csv")
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    t_1pct = E.threshold_at_fpr(y_calib, s_calib, E.ONE_PCT_FPR)
    test = pd.read_csv("data/splits-p3/test.csv")
    urls = test["url"].astype(str).tolist()
    y = test["label"].to_numpy(dtype=int)
    _, s_test = score_band("data/splits-p3/test.csv")
    tier1 = np.asarray(s_test, dtype=float)
    tier1_by_url = dict(zip(urls, [float(v) for v in tier1], strict=True))

    extracts: dict[str, dict] = {}
    for line in open("data/snapshots-p4/results.jsonl", encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if "extract" in row:
                extracts[row["url"]] = json.loads(row["extract"])
    baseline = np.array(PasswordBaseline(extracts).score(urls), dtype=float)

    report: dict = {
        "run_id": args.run_id,
        "run_class": run_class,
        "n_repeats": n_repeats,
        # Verdicts present (non-None). A 400 schema refusal is a valid seal
        # that retains Tier-1, so this sits below run.json's seal coverage.
        "verdict_coverage_per_repeat": coverage,
        "t_alert": t_alert,
        "lower_edge": lower_edge,
        "t_1pct": float(t_1pct),
        "n_test": len(urls),
    }

    per_repeat: list[dict[str, object]] = []
    for policy in ("default", "alternative"):
        repeat_blocks = [
            _block_for_repeat(
                verdicts_by_repeat[r],
                urls,
                y,
                tier1,
                baseline,
                tier1_by_url,
                t_alert,
                lower_edge,
                float(t_1pct),
                policy,
            )
            for r in repeat_indexes
        ]
        block: dict = {"policy": policy}
        for label, _ in LABELS:
            cascade_recalls = [
                float(b[label]["cascade"]["recall"]) for b in repeat_blocks
            ]
            cascade_fprs = [float(b[label]["cascade"]["fpr"]) for b in repeat_blocks]
            lifts = [
                (
                    float(b[label]["paired_pr_auc_lift_ci"][0]),
                    float(b[label]["paired_pr_auc_lift_ci"][1]),
                )
                for b in repeat_blocks
            ]
            all_indist = all(bool(b[label]["indistinguishable"]) for b in repeat_blocks)
            block[label] = {
                "cascade": {
                    "recall": _median(cascade_recalls),
                    "recall_range": _minmax(cascade_recalls),
                    "fpr": _median(cascade_fprs),
                    "fpr_range": _minmax(cascade_fprs),
                    "tp": median_int(
                        [int(b[label]["cascade"]["tp"]) for b in repeat_blocks]
                    ),
                    "fp": median_int(
                        [int(b[label]["cascade"]["fp"]) for b in repeat_blocks]
                    ),
                },
                "tier1": repeat_blocks[0][label]["tier1"],
                "password_baseline": repeat_blocks[0][label]["password_baseline"],
                "paired_pr_auc_lift_ci": [
                    min(lo for lo, _ in lifts),
                    max(hi for _, hi in lifts),
                ],
                "per_repeat_cis": [[lo, hi] for lo, hi in lifts],
                "indistinguishable": all_indist,
                "verdict": "indistinguishable" if all_indist else "distinguished",
            }
            for r, b in zip(repeat_indexes, repeat_blocks, strict=True):
                per_repeat.append(
                    {"repeat_idx": r, "policy": policy, "label": label, **(b[label])}
                )
        report[policy] = block

    report["per_repeat"] = per_repeat

    # Descriptive: agreement + verdict distribution, pooled over repeats.
    manifest = pd.DataFrame(
        json.loads(Path("reports/snapshot-manifest-p4.json").read_text())["rows"]
    )
    ok_urls = set(manifest.loc[manifest["outcome"] == "ok", "url"].astype(str))
    labels = dict(zip(urls, [int(v) for v in y], strict=True))
    agree_pairs = 0
    agree_total = 0
    dist: Counter[str] = Counter()
    for r in repeat_indexes:
        for u, v in verdicts_by_repeat[r].items():
            if v is None:
                continue
            agree_total += 1
            agree_pairs += int((v == "phishing") == (labels.get(u) == 1))
            dist[v] += 1
    report["descriptive"] = {
        "agreement_sealed": agree_pairs / max(1, agree_total),
        "n_sealed": agree_total,
        "verdict_distribution": dict(dist),
        "n_test_fetched_ok": len(ok_urls & set(test["url"].astype(str))),
    }

    # Cost + latency from this run's cold-cache seals.
    usages: list[dict] = []
    latencies: list[float] = []
    run_file = run_dir / "judgments.jsonl"
    if run_file.exists():
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

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = build_markdown(args.run_id, run_class, n_repeats, coverage, report)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def median_int(values: list[int]) -> int:
    return int(round(float(np.median(np.asarray(values, dtype=float)))))


def build_markdown(
    run_id: str,
    run_class: str,
    n_repeats: int,
    coverage: dict[str, int],
    report: dict,
) -> list[str]:
    recorded = run_class == "recorded"
    if recorded:
        lines = [
            f"# Phase 4 report — LLM layer (recorded run `{run_id}`)",
            "",
            f"Run class `recorded`: {n_repeats} full-population repeats, "
            f"coverage {coverage}. The headline is a **range over repeats** "
            "(criterion 10b) because determinism measured 22% > 5%.",
            "Coverage is stated beside every number (criterion 14).",
            "",
        ]
    else:
        lines = [
            f"# Phase 4 report — provisional run `{run_id}`",
            "",
            f"Run class `provisional` (coverage {coverage}): no number here "
            "publishes. Coverage is stated beside every number (criterion 14).",
            "",
        ]
    for policy in ("default", "alternative"):
        lines.append(f"## Cascade [{policy}] ({run_class})")
        for label, _ in LABELS:
            b = report[policy][label]
            lines.append(
                f"- {label}: cascade recall="
                f"{b['cascade']['recall']:.4f} "
                f"[{b['cascade']['recall_range'][0]:.4f}, "
                f"{b['cascade']['recall_range'][1]:.4f}] "
                f"fpr={b['cascade']['fpr']:.5f} "
                f"[{b['cascade']['fpr_range'][0]:.5f}, "
                f"{b['cascade']['fpr_range'][1]:.5f}] "
                f"vs tier1 recall={b['tier1']['recall']:.4f} "
                f"fpr={b['tier1']['fpr']:.5f} vs password baseline "
                f"recall={b['password_baseline']['recall']:.4f} "
                f"fpr={b['password_baseline']['fpr']:.5f}; "
                f"paired PR-AUC lift "
                f"[{b['paired_pr_auc_lift_ci'][0]:.4f}, "
                f"{b['paired_pr_auc_lift_ci'][1]:.4f}] -> {b['verdict']}."
            )
    desc = report["descriptive"]
    cost = report["cost"]
    lines += [
        "",
        f"## Descriptive ({run_class})",
        f"- agreement on sealed rows: {desc['agreement_sealed']:.3f} "
        f"(n={desc['n_sealed']})",
        f"- verdict distribution: {desc['verdict_distribution']}",
        f"- cost (cold-cache, n={cost['n_cold_calls']}): prompt "
        f"{cost['prompt_tokens_mean']:.2f} / completion "
        f"{cost['completion_tokens_mean']:.2f} tokens per call; latency p50 "
        f"{cost['latency_p50_ms']:.0f}ms p90 {cost['latency_p90_ms']:.0f}ms; "
        f"forecast ${cost['forecast_usd_per_call']:.5f}/call at $0.15/$0.60 "
        "per 1M (Groq listed, 2026-09-18).",
        f"- verdict coverage (non-None) per repeat: {coverage} of 1,106 "
        "in-band fetched-ok rows; 400 schema refusals are valid seals that "
        "retain Tier-1 (§2 failure policy).",
        "",
        "> Prereg criteria (9b/10b and the phase question) are closed out in "
        "`docs/phase4-preregistration.md` and the model card; this report "
        "carries the numbers only."
        if recorded
        else "> This is the provisional close-out; the recorded-run criteria are "
        "reported by the close-out that owns them.",
        "",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
