"""Pre-LLM reference (§7): password baseline beside Tier-1, before any LLM
output is read.

Scores the full test band with `PasswordBaseline` (pure function of the
frozen extracts; missing snapshot → 0.0) and with Tier-1 row (a), reporting
recall/FPR at the calib-fixed `t_alert`. Sealed to
`reports/phase4-baseline-ref.json`. No LLM call, no verdict read — this is
the comparison the LLM layer must beat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import eval as E  # noqa: E402
from phishnet.llm.password_baseline import PasswordBaseline  # noqa: E402
from phishnet.snapshot.tier1 import band_edges, score_band  # noqa: E402

OUT = Path("reports/phase4-baseline-ref.json")


def main() -> int:
    extracts: dict[str, dict] = {}
    with open("data/snapshots-p4/results.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "extract" in row:
                    extracts[row["url"]] = json.loads(row["extract"])

    test = pd.read_csv("data/splits-p3/test.csv", usecols=["url", "label"])
    urls = test["url"].astype(str).tolist()
    y = test["label"].to_numpy(dtype=int)

    baseline = np.array(PasswordBaseline(extracts).score(urls), dtype=float)
    y_calib, s_calib = score_band("data/splits-p3/calib.csv")
    t_alert, lower_edge = band_edges(y_calib, s_calib)
    _, s_test = score_band("data/splits-p3/test.csv")
    tier1 = np.asarray(s_test, dtype=float)

    ref: dict = {
        "t_alert": t_alert,
        "lower_edge": lower_edge,
        "n_test": int(y.size),
        "n_extracts": len(extracts),
    }
    for name, scores in (("tier1", tier1), ("password_baseline", baseline)):
        conf = E.confusion_at(y, scores, t_alert)
        rates = E.rates_at(y, scores, t_alert)
        ref[name] = {
            "recall": rates["recall"],
            "fpr": rates["fpr"],
            "tp": conf["tp"],
            "fp": conf["fp"],
            "tn": conf["tn"],
            "fn": conf["fn"],
        }
        print(
            f"{name}: recall={rates['recall']:.4f} "
            f"fpr={rates['fpr']:.5f} tp={conf['tp']} fp={conf['fp']}"
        )
    fires = int((baseline == 1.0).sum())
    ref["password_baseline"]["fires"] = fires
    OUT.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"sealed -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
