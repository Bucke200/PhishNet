"""Live-performance eval harness (Phase A).

Pushes a labeled URL fixture through the **serving** path in-process and
reports recall/FPR plus a per-URL band decomposition (`below` / `in` /
`above`) and the Tier-2 outcome. This is the measurement that decides whether
live failures are below-band false negatives, above-band false positives, or
sealed-mode `can't assess` noise — before any model change.

No network by default:
  * the shortener resolver runs only with ``--resolve``;
  * Tier 2 is disabled unless ``--tier2 sealed|live`` is passed.

The first measurement must isolate Tier 1, so the defaults are deliberately
offline.

Usage:
  uv run python scripts/live_eval.py
  uv run python scripts/live_eval.py --tier2 sealed
  uv run python scripts/live_eval.py --tier2 sealed --floor 0.3
  uv run python scripts/live_eval.py --tier2 sealed --mapping strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from phishnet.enrichment.key import host_of, registrable_domain  # noqa: E402
from phishnet.serving.app import predict_one  # noqa: E402
from phishnet.serving.cascade import ALERT, CANT_ASSESS  # noqa: E402
from phishnet.serving.shortener import resolve  # noqa: E402
from phishnet.serving.tier1 import Tier1Servable  # noqa: E402

DEFAULT_FIXTURE = Path("tests/fixtures/live-labeled.csv")
TRAIN = Path("data/splits-p3/train.csv")


def build_tier2(mode: str):  # noqa: ANN201 - provider is a protocol
    if mode == "off":
        return None
    if mode == "sealed":
        from phishnet.serving.tier2 import SealedTier2Provider

        try:
            return SealedTier2Provider()
        except FileNotFoundError as exc:
            raise SystemExit(f"sealed Tier-2 unavailable: {exc}") from exc
    if mode == "live":
        import os

        from phishnet.serving.tier2 import LiveTier2Provider

        key = os.environ.get("GROQ_API_KEY", "")
        fetcher = os.environ.get("PHISHNET_FETCHER_URL", "")
        if not key or not fetcher:
            raise SystemExit("live Tier-2 needs GROQ_API_KEY and PHISHNET_FETCHER_URL")
        return LiveTier2Provider(fetcher, key)
    raise SystemExit(f"unknown --tier2 mode: {mode}")


def apply_policy(
    disposition: str,
    reason: str,
    tier1_score: float | None,
    *,
    mapping: str,
    strict_floor: float,
) -> tuple[str, str]:
    """Post-process an alert under a stricter mapping (Phase A experiments)."""
    if disposition != ALERT:
        return disposition, reason
    low = tier1_score is None or tier1_score < strict_floor
    if mapping == "strict" and reason.startswith("tier2_phishing") and low:
        return CANT_ASSESS, "strict_downgrade"
    return disposition, reason


def band_of(tier1_score: float | None, floor: float, t_alert: float) -> str:
    if tier1_score is None:
        return "unscored"
    if tier1_score >= t_alert:
        return "above"
    if tier1_score < floor:
        return "below"
    return "in"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        default=None,
        help="labeled CSV (repeatable); default tests/fixtures/live-labeled.csv",
    )
    parser.add_argument("--tier2", choices=["off", "sealed", "live"], default="off")
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument(
        "--mapping", choices=["registered", "strict"], default="registered"
    )
    parser.add_argument(
        "--failure-policy",
        choices=["closed", "graded", "mechanism"],
        default="closed",
        help="failure disposition: closed (registered) | graded | mechanism",
    )
    parser.add_argument(
        "--failure-floor",
        type=float,
        default=0.85,
        help="Tier-1 score at/above which a Tier-2 failure still alerts "
        "(only when --failure-policy graded)",
    )
    parser.add_argument("--strict-floor", type=float, default=0.8)
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    fixtures = args.fixture or [DEFAULT_FIXTURE]
    frame = pd.concat([pd.read_csv(f) for f in fixtures], ignore_index=True)
    tier1 = Tier1Servable()
    t_alert = tier1.thresholds["t_alert"]
    lower_edge = tier1.thresholds["lower_edge"]
    floor = lower_edge if args.floor is None else args.floor
    failure_floor = args.failure_floor if args.failure_policy == "graded" else None
    tier2 = build_tier2(args.tier2)
    train_domains: set[str] = set()
    if TRAIN.is_file():
        train = pd.read_csv(TRAIN, usecols=["registrable_domain"])
        train_domains = set(train["registrable_domain"].astype(str).str.lower())

    rows: list[dict] = []
    for _, record in frame.iterrows():
        url = str(record["url"])
        body = predict_one(
            url,
            tier1=tier1,
            resolver=resolve if args.resolve else None,
            tier2=tier2,
            t_alert=t_alert,
            lower_edge=lower_edge,
            tier2_floor=floor,
            tier2_failure_floor=failure_floor,
            tier2_failure_policy=args.failure_policy,
        )
        tier1_score = body["tier1_score"]
        disposition, reason = apply_policy(
            str(body["disposition"]),
            str(body["reason"]),
            tier1_score,
            mapping=args.mapping,
            strict_floor=args.strict_floor,
        )
        host = host_of(url)
        rows.append(
            {
                "url": url,
                "label": int(record["label"]),
                "source": str(record["source"]),
                "tier1_score": tier1_score,
                "band": band_of(tier1_score, floor, t_alert),
                "disposition": disposition,
                "reason": reason,
                "tier2": body["tier2"],
                "scored_url": body["scored_url"],
                "host": host,
                "domain": registrable_domain(host) if host else "",
                "domain_in_train": (registrable_domain(host).lower() in train_domains)
                if host
                else False,
            }
        )

    def metrics(subset: list[dict]) -> dict:
        tp = sum(1 for r in subset if r["label"] == 1 and r["disposition"] == ALERT)
        fn = sum(1 for r in subset if r["label"] == 1 and r["disposition"] != ALERT)
        fp = sum(1 for r in subset if r["label"] == 0 and r["disposition"] == ALERT)
        tn = sum(1 for r in subset if r["label"] == 0 and r["disposition"] != ALERT)
        ca = sum(1 for r in subset if r["disposition"] == CANT_ASSESS)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "n": len(subset),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "cant_assess": ca,
            "recall": recall,
            "fpr": fpr,
            "alerts": tp + fp,
        }

    overall = metrics(rows)
    by_band = Counter((r["label"], r["band"]) for r in rows)
    by_reason = Counter(r["reason"] for r in rows if r["disposition"] == ALERT)
    by_host = Counter(r["domain"] for r in rows)
    dup_domains = {d: c for d, c in by_host.items() if c > 1 and d}

    scorable = [r for r in rows if r["tier1_score"] is not None]
    scored_metrics = metrics(scorable)
    leak = [r for r in rows if r["domain_in_train"]]
    leak_metrics = metrics(leak) if leak else None

    print(f"fixture: {[str(f) for f in fixtures]}  rows={len(rows)}")
    print(f"tier1: t_alert={t_alert:.6f} lower_edge={lower_edge:.6f} floor={floor:.6f}")
    print(
        f"tier2={args.tier2} mapping={args.mapping} "
        f"failure_policy={args.failure_policy} failure_floor={failure_floor} "
        f"resolve={args.resolve}"
    )
    print()
    print("== per URL ==")
    for r in rows:
        score = "n/a" if r["tier1_score"] is None else f"{r['tier1_score']:.4f}"
        t2 = "" if r["tier2"] is None else f" tier2={r['tier2']['kind']}"
        print(
            f"  label={r['label']} score={score:>6} band={r['band']:<8} "
            f"disp={r['disposition']:<12} reason={r['reason']}{t2}  {r['url']}"
        )
    print()
    print("== metrics ==")
    by_source = {
        source: metrics([r for r in rows if r["source"] == source])
        for source in sorted({r["source"] for r in rows})
    }
    print(f"  overall: {overall}")
    print(f"  scorable (tier1 present): {scored_metrics}")
    for source, source_metrics in by_source.items():
        print(f"  source {source}: {source_metrics}")
    if leak_metrics is not None:
        print(f"  domain present in train ({len(leak)} rows): {leak_metrics}")
    print(f"  label x band: {dict(by_band)}")
    print(f"  alert reasons: {dict(by_reason)}")
    if dup_domains:
        print(f"  repeated domains: {dup_domains}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "fixture": str(args.fixture),
                    "config": {
                        "tier2": args.tier2,
                        "mapping": args.mapping,
                        "failure_policy": args.failure_policy,
                        "failure_floor": failure_floor,
                        "floor": floor,
                        "resolve": args.resolve,
                        "strict_floor": args.strict_floor,
                    },
                    "thresholds": {"t_alert": t_alert, "lower_edge": lower_edge},
                    "overall": overall,
                    "scorable": scored_metrics,
                    "by_source": by_source,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
