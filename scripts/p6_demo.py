"""Phase 6 demo driver (C6): three scenarios, sealed Tier-2 by default.

Runs a benign page, a phishing page, and a detector-escalated injection page
through the serving pipeline and prints the disposition, Tier-1 score, and
top-3 native SHAP contributions. Default is in-process against the sealed
Phase 5 verdicts; `--base URL` targets a running container instead.

The browser recording is an operator step (screen-capture the extension
against the container); this script produces the reproducible transcript that
the recording demonstrates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from phishnet.serving import Tier1Servable
from phishnet.serving.app import predict_one
from phishnet.serving.tier2 import SealedTier2Provider

MANIFEST = Path("reports/adversarial-manifest-p5.json")
OUT = Path("reports/phase6-demo.json")


def select_scenarios() -> list[dict[str, str]]:
    provider = SealedTier2Provider()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chosen: dict[str, str] = {}
    for row in manifest:
        outcome = provider.judge(row["url"])
        if outcome is None:
            continue
        page_id = row["page_id"]
        if (
            "benign" not in chosen
            and page_id.startswith("clean-benign")
            and outcome.kind == "benign"
        ):
            chosen["benign"] = row["url"]
        if (
            "phishing" not in chosen
            and page_id.startswith("clean-phish")
            and outcome.kind == "phishing"
            and outcome.reason != "detector"
        ):
            chosen["phishing"] = row["url"]
        if (
            "injection" not in chosen
            and row.get("detector_hit")
            and outcome.reason == "detector"
        ):
            chosen["injection"] = row["url"]
    missing = {"benign", "phishing", "injection"} - set(chosen)
    if missing:
        raise SystemExit(f"could not find demo scenarios: {sorted(missing)}")
    return [{"label": label, "url": url} for label, url in chosen.items()]


def run_http(
    base: str, scenarios: list[dict[str, str]], *, allow_live: bool = False
) -> list[dict[str, Any]]:
    import requests

    # The registered transcript is a sealed replay of the frozen Phase 5
    # verdicts. A live-mode container fetches the current pages, which have
    # changed since Phase 5 (the "phishing" page is now benign; the "benign"
    # redirect page trips the detector), so it would produce a transcript that
    # contradicts the demo. Refuse unless the operator opts in.
    health = requests.get(f"{base}/health", timeout=30).json()
    mode = health.get("tier2_mode")
    print(f"container tier2_mode: {mode}")
    if mode != "sealed" and not allow_live:
        raise SystemExit(
            f"demo transcript is registered against sealed Tier 2, but the "
            f"container is '{mode}'. Restart it without PHISHNET_TIER2_MODE=live "
            f"(the image defaults to sealed), or pass --allow-live to record a "
            f"live demo knowing the labels will not match."
        )

    results = []
    for scenario in scenarios:
        response = requests.post(
            f"{base}/predict", json={"url": scenario["url"]}, timeout=30
        )
        response.raise_for_status()
        body = response.json()
        attribution = None
        if body.get("in_band") and body.get("score") is not None:
            explained = requests.post(
                f"{base}/explain",
                json={"url": scenario["url"], "top_k": 3},
                timeout=30,
            )
            if explained.ok:
                attribution = explained.json()["attribution"]
        results.append({**scenario, **body, "attribution": attribution})
    return results


def run_in_process(scenarios: list[dict[str, str]]) -> list[dict[str, Any]]:
    tier1 = Tier1Servable()
    tier2 = SealedTier2Provider()
    results = []
    for scenario in scenarios:
        body = predict_one(
            scenario["url"],
            tier1=tier1,
            resolver=None,
            tier2=tier2,
            t_alert=tier1.thresholds["t_alert"],
            lower_edge=tier1.thresholds["lower_edge"],
        )
        attribution = None
        if body["in_band"] and body["score"] is not None:
            attribution = tier1.explain_one(scenario["url"], top_k=3)
        results.append({**scenario, **body, "attribution": attribution})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="running container base URL")
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="allow recording against a live-mode container (labels will differ)",
    )
    args = parser.parse_args(argv)

    scenarios = select_scenarios()
    results = (
        run_http(args.base, scenarios, allow_live=args.allow_live)
        if args.base
        else run_in_process(scenarios)
    )
    for result in results:
        print(f"=== {result['label']}: {result['url']}")
        print(
            f"    disposition={result['disposition']} "
            f"score={result['score']} tier1={result['tier1_score']:.6f} "
            f"mode={result.get('tier2_mode')} reason={result['reason']}"
        )
        if result.get("tier2"):
            print(f"    tier2={result['tier2']}")
        if result.get("attribution"):
            for feature in result["attribution"]["features"]:
                print(f"      {feature['feature']}: {feature['contribution']:.4f}")
    OUT.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
