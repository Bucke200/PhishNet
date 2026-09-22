"""Build the expanded benign arm for live evaluation (Phase A).

Samples popular-but-not-training domains from the pinned Tranco list across
three rank bands (head / mid / tail), excludes every domain that appears in
the Phase 3 training split (so the model cannot have memorized it), and adds
a small set of `/login` paths to exercise the in-band login false-alarm risk.

Deterministic: seed 0, no network.

Usage:
  uv run python scripts/build_live_benign.py
  uv run python scripts/build_live_benign.py --n-head 100 --n-mid 150 --n-tail 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

TRANCO = Path("data/raw/tranco-46VQX-top1000000-2026-09-13.csv")
TRAIN = Path("data/splits-p3/train.csv")
EXISTING = Path("tests/fixtures/live-labeled.csv")
OUT = Path("tests/fixtures/live-benign-expanded.csv")

BANDS = {
    "tranco-head": (1, 1_000),
    "tranco-mid": (100_000, 250_000),
    "tranco-tail": (600_000, 1_000_000),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tranco", type=Path, default=TRANCO)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--existing", type=Path, default=EXISTING)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-head", type=int, default=100)
    parser.add_argument("--n-mid", type=int, default=150)
    parser.add_argument("--n-tail", type=int, default=200)
    parser.add_argument("--n-login", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tranco = pd.read_csv(args.tranco, header=None, names=["rank", "domain"])
    train = set(
        pd.read_csv(args.train, usecols=["registrable_domain"])["registrable_domain"]
        .astype(str)
        .str.lower()
    )
    existing_domains: set[str] = set()
    if args.existing.is_file():
        ex = pd.read_csv(args.existing)
        for url in ex["url"].astype(str):
            host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
            existing_domains.add(host.removeprefix("www."))

    excluded = train | existing_domains
    rng = np.random.default_rng(args.seed)

    def sample(lo: int, hi: int, n: int) -> list[str]:
        pool = (
            tranco.loc[(tranco["rank"] >= lo) & (tranco["rank"] <= hi), "domain"]
            .astype(str)
            .str.lower()
            .tolist()
        )
        pool = [d for d in pool if d not in excluded]
        arr = np.array(pool, dtype=object)
        rng.shuffle(arr)
        return arr[:n].tolist()

    rows: list[dict[str, object]] = []
    counts = {
        "tranco-head": args.n_head,
        "tranco-mid": args.n_mid,
        "tranco-tail": args.n_tail,
    }
    chosen: list[str] = []
    for source, n in counts.items():
        lo, hi = BANDS[source]
        for domain in sample(lo, hi, n):
            chosen.append(domain)
            rows.append(
                {
                    "url": f"https://{domain}/",
                    "label": 0,
                    "source": source,
                    "first_seen": "",
                    "domain_in_train": domain in train,
                    "extension_disposition": "",
                    "note": f"tranco rank band {lo}-{hi}",
                }
            )

    login_pool = sample(1, 250_000, args.n_login)
    for domain in login_pool:
        rows.append(
            {
                "url": f"https://{domain}/login",
                "label": 0,
                "source": "tranco-login",
                "first_seen": "",
                "domain_in_train": domain in train,
                "extension_disposition": "",
                "note": "login-path false-alarm probe",
            }
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} benign rows -> {args.out}")
    print(out["source"].value_counts().to_dict())
    print(
        f"excluded {len(train)} train domains + {len(existing_domains)} fixture hosts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
