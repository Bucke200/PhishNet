"""Step 4: enriched feature tables (lexical + age/CT, one vocabulary).

Turns joined split rows into train-ready matrices: the shared lexical
pipeline (``featurise_frame``, scheme switch included) plus the enriched
columns, in one fixed column order. The vocabulary (lexical + enriched)
is returned alongside the frame so training persists exactly what
scoring must load — never a parallel column list.

Representation (pre-committed):

* values are floats; ``*_known`` flags ride beside them. Unknown reads
  NaN (LightGBM handles NaN natively and learns which way to send it) —
  never 0.0, which collides with real zeros: domain_age_days=0 means
  "registered today" and ct_cert_count_pre=0 with ct_known means
  "looked, nothing pre-cutoff", both genuine answers distinct from
  missing;
* ``age_na``/``ct_na`` are EXCLUDED from features (they stay in joined
  rows for gating and analysis). Hosted-ness instead enters X
  explicitly as ``is_hosted_tenant`` (0.0/1.0, always known — URL-
  derived from the PSL-private/vendor list, serving-time known, no
  time dependence): hiding it while hosted rows read known=0 lets the
  model learn "not known → phishing" from tenancy, and the enrichment
  lift would pick up the hosted indicator. With the flag explicit, a
  known=0 on a non-hosted row means a failed lookup and nothing else.
  The flag rides EVERY row including lexical-only (Amendment A).
* ``ct_known`` with count 0 is a real answer (looked, nothing
  pre-cutoff): flag 1.0, count 0.0, age 0.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from phishnet.enrichment.join import join_enrichment
from phishnet.enrichment.key import host_of, is_hosted_tenant
from phishnet.features.extraction import featurise_frame

ENRICHED_COLUMNS: list[str] = [
    "domain_age_days",
    "age_known",
    "ct_age_days",
    "ct_cert_count_pre",
    "ct_known",
]

# Lexical-kind flag, present in EVERY ablation row including lexical-only:
# URL-derived, serving-time known, no time dependence, never missing.
HOSTED_COLUMN = "is_hosted_tenant"


def hosted_flag(urls: list[str]) -> list[float]:
    """Explicit hosted-tenancy indicator (1.0/0.0, always known)."""
    return [1.0 if is_hosted_tenant(host_of(str(u))) else 0.0 for u in urls]


def enriched_row_features(row: dict[str, Any]) -> dict[str, float]:
    """The five feature values for one joined row (na flags excluded)."""
    nan = float("nan")
    out: dict[str, float] = {
        "domain_age_days": nan,
        "age_known": 0.0,
        "ct_age_days": nan,
        "ct_cert_count_pre": nan,
        "ct_known": 0.0,
    }
    if row.get("age_known"):
        out["domain_age_days"] = float(row["domain_age_days"])
        out["age_known"] = 1.0
    if row.get("ct_known"):
        out["ct_age_days"] = (
            float(row["ct_age_days"]) if row.get("ct_age_days") is not None else nan
        )
        out["ct_cert_count_pre"] = float(row.get("ct_cert_count_pre") or 0.0)
        out["ct_known"] = 1.0
    return out


def build_feature_table(
    rows: list[dict[str, Any]],
    snapshot: Path,
    selection: dict[str, Any],
    lexical_columns: list[str],
    *,
    canonicalize: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Join + featurise: (X, vocabulary, manifest-fragment).

    ``rows`` carry url/label/first_seen/survival_stratum (see
    ``join_enrichment``); ``selection`` is exactly one rule (pinned run
    or earliest-success). Column order is lexical, hosted flag, then
    enriched — the returned vocabulary IS the contract training persists
    and scoring loads.
    """
    joined, join_manifest = join_enrichment(rows, snapshot, selection)
    urls = [str(r["url"]) for r in joined]
    lex = featurise_frame(
        urls,
        lexical_columns,
        canonicalize=canonicalize,
    ).reset_index(drop=True)
    host = pd.DataFrame({HOSTED_COLUMN: hosted_flag(urls)})
    enr = pd.DataFrame(
        [enriched_row_features(r) for r in joined], columns=ENRICHED_COLUMNS
    )
    frame = pd.concat([lex, host, enr], axis=1)
    vocabulary = [*lexical_columns, HOSTED_COLUMN, *ENRICHED_COLUMNS]
    assert list(frame.columns) == vocabulary
    manifest = {
        "join": join_manifest,
        "vocabulary": vocabulary,
        "hosted_column": HOSTED_COLUMN,
        "n_enriched_columns": len(ENRICHED_COLUMNS),
        "canonicalize": canonicalize,
    }
    return frame, vocabulary, manifest


def apply_miss(
    frame: pd.DataFrame,
    keys: list[str],
    miss_fraction: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Cold-start simulation: force enriched features to unknown.

    Misses group by CACHE KEY (every enriched field for a key goes
    unknown together, flags false) — the same rule as the serving stub,
    so the 100% row equals stub behavior by contract (pinned by test).
    ``keys`` parallels the frame rows; ``miss_fraction`` of distinct keys
    is drawn with ``seed``. 0.0 returns the frame unchanged. The hosted
    flag is URL-derived, not lookup-dependent, so misses never touch it.
    """
    if not 0.0 <= miss_fraction <= 1.0:
        raise ValueError(f"miss_fraction must be in [0, 1], got {miss_fraction}")
    out = frame.copy()
    if miss_fraction == 0.0:
        return out
    import numpy as np

    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    n_miss = int(round(miss_fraction * len(uniq)))
    missed = set(rng.choice(uniq, n_miss, replace=False).tolist()) if n_miss else set()
    miss_rows = [k in missed for k in keys]
    # Missed rows read exactly like failed lookups (NaN + flags 0).
    out.loc[miss_rows, ENRICHED_COLUMNS] = [
        float("nan"),
        0.0,
        float("nan"),
        float("nan"),
        0.0,
    ]
    return out
