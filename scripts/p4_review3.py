"""Verify close-out review items 2–3 (read-only over seals)."""

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

# Item 2: fetched denominators, by class.
fetched_ok = manifest.loc[manifest["outcome"] == "ok", "url"].astype(str).tolist()
fmask = test["url"].astype(str).isin(set(fetched_ok)).to_numpy()
for name, cls in (("phish", 1), ("benign", 0)):
    m = fmask & (y == cls)
    print(
        f"fetched-ok test {name}: n={m.sum()} fires={int((baseline[m] == 1.0).sum())}"
    )

# Item 3: NaN provenance — era split, and NaN <=> step0-sample membership.
print("NaN tier1 by era:")
print(manifest[manifest["tier1_score"].isna()].groupby("era").size().to_string())
print("total NaN:", int(manifest["tier1_score"].isna().sum()))

# The 5: test-era, ok, NaN rows — confirm test-split membership via splits.
nan_test_ok = manifest[
    (manifest["era"] == "test")
    & (manifest["outcome"] == "ok")
    & (manifest["tier1_score"].isna())
]
test_urls = set(test["url"].astype(str))
calib_urls = set(
    pd.read_csv("data/splits-p3/calib.csv", usecols=["url"])["url"].astype(str)
)
train_urls = set(
    pd.read_csv("data/splits-p3/train.csv", usecols=["url"])["url"].astype(str)
)
in_test = sum(u in test_urls for u in nan_test_ok["url"].astype(str))
in_calib = sum(u in calib_urls for u in nan_test_ok["url"].astype(str))
in_train = sum(u in train_urls for u in nan_test_ok["url"].astype(str))
print(
    f"5-row check: n={len(nan_test_ok)} in_test={in_test} "
    f"in_calib={in_calib} in_train={in_train}"
)
