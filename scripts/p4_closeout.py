"""Close-out analysis (read-only): baseline fires, structural ceiling, tokens.

No network, no run-store writes. Reads only sealed artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from phishnet.llm.password_baseline import PasswordBaseline  # noqa: E402

manifest = pd.DataFrame(
    json.loads(Path("reports/snapshot-manifest-p4.json").read_text())["rows"]
)
extracts: dict[str, dict] = {}
for line in Path("data/snapshots-p4/results.jsonl").read_text().splitlines():
    if line.strip():
        row = json.loads(line)
        if "extract" in row:
            extracts[row["url"]] = json.loads(row["extract"])

test = pd.read_csv("data/splits-p3/test.csv")
urls = test["url"].astype(str).tolist()
y = test["label"].to_numpy(dtype=int)
baseline = np.array(PasswordBaseline(extracts).score(urls))
man_idx = manifest.set_index("url")

# Baseline fires: overall + among fetched, by class.
ok_urls = set(manifest.loc[manifest["outcome"] == "ok", "url"].astype(str))
fetched = test["url"].astype(str).isin(ok_urls)
for name, mask in (
    ("all-test", np.ones(len(test), bool)),
    ("fetched-ok-test", fetched.to_numpy()),
):
    sub_y = y[mask]
    sub_b = baseline[mask]
    print(
        f"{name}: n={mask.sum()} fires={(sub_b == 1.0).sum()} "
        f"phish-fires={int(((sub_b == 1.0) & (sub_y == 1)).sum())} "
        f"benign-fires={int(((sub_b == 1.0) & (sub_y == 0)).sum())}"
    )

# Structural ceiling from sealed manifest (test band, in-band edges).
T_ALERT, LOWER = 0.9269363298832987, 0.6493076453312958
t = manifest[manifest["era"] == "test"].copy()
t["in_band"] = (t["tier1_score"] >= LOWER) & (t["tier1_score"] < T_ALERT)
tp_total = int((test["label"] == 1).sum())
tb_total = int((test["label"] == 0).sum())
inband_ok = t[t["in_band"] & (t["outcome"] == "ok")]
ph = int((inband_ok["label"] == 1).sum())
be = int((inband_ok["label"] == 0).sum())
print(f"ceiling: fetched in-band test phish {ph}/{tp_total} = {ph / tp_total:.4f}")
print(f"exposure: fetched in-band test benign {be}/{tb_total} = {be / tb_total:.4f}")
print(f"1106 check: in-band test fetched-ok in manifest = {len(inband_ok)}")

# Exact token means from cold-cache seals.
usages = []
for run_file in Path("runs/phase4").glob("*/judgments.jsonl"):
    for line in run_file.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("cold_cache") and row.get("usage"):
                usages.append(row["usage"])
pt = np.array([u.get("prompt_tokens", 0) for u in usages], float)
ct = np.array([u.get("completion_tokens", 0) for u in usages], float)
rt = np.array(
    [u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) for u in usages],
    float,
)
vis = ct - rt
print(
    f"n={len(usages)} prompt={pt.mean():.2f} completion={ct.mean():.2f} "
    f"reasoning={rt.mean():.2f} visible={vis.mean():.2f}"
)
print(
    f"sum: reasoning+visible={rt.mean() + vis.mean():.2f} vs completion={ct.mean():.2f}"
)
