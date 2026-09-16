"""Step 4: enriched feature tables (lexical + age/CT, one vocabulary).

Turns joined split rows into train-ready matrices: the shared lexical
pipeline (``featurise_frame``, scheme switch included) plus the enriched
columns, in one fixed column order. The vocabulary (lexical + enriched)
is returned alongside the frame so training persists exactly what
scoring must load — never a parallel column list.

Representation (pre-committed):

* values are floats; ``*_known`` flags ride beside them (unknown → value
  0.0, flag 0.0 — the model sees "no information", never a magic number
  that could read as young/old);
* ``age_na``/``ct_na`` are EXCLUDED from features. They mark hosted
  tenants, and hosted-tenancy leans phishing — a na flag would smuggle
  that correlation in as a learned feature. They stay in the joined
  rows for gating and analysis, never in X.
* ``ct_known`` with count 0 is a real answer (looked, nothing
  pre-cutoff): flag 1.0, count 0.0, age 0.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from phishnet.enrichment.join import join_enrichment
from phishnet.features.extraction import featurise_frame

ENRICHED_COLUMNS: list[str] = [
    "domain_age_days",
    "age_known",
    "ct_age_days",
    "ct_cert_count_pre",
    "ct_known",
]


def enriched_row_features(row: dict[str, Any]) -> dict[str, float]:
    """The five feature values for one joined row (na flags excluded)."""
    out: dict[str, float] = {
        "domain_age_days": 0.0,
        "age_known": 0.0,
        "ct_age_days": 0.0,
        "ct_cert_count_pre": 0.0,
        "ct_known": 0.0,
    }
    if row.get("age_known"):
        out["domain_age_days"] = float(row.get("domain_age_days") or 0.0)
        out["age_known"] = 1.0
    if row.get("ct_known"):
        out["ct_age_days"] = float(row.get("ct_age_days") or 0.0)
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
    or earliest-success). Column order is lexical then enriched — the
    returned vocabulary IS the contract training persists and scoring
    loads.
    """
    joined, join_manifest = join_enrichment(rows, snapshot, selection)
    lex = featurise_frame(
        [str(r["url"]) for r in joined],
        lexical_columns,
        canonicalize=canonicalize,
    ).reset_index(drop=True)
    enr = pd.DataFrame(
        [enriched_row_features(r) for r in joined], columns=ENRICHED_COLUMNS
    )
    frame = pd.concat([lex, enr], axis=1)
    vocabulary = [*lexical_columns, *ENRICHED_COLUMNS]
    assert list(frame.columns) == vocabulary
    manifest = {
        "join": join_manifest,
        "vocabulary": vocabulary,
        "n_enriched_columns": len(ENRICHED_COLUMNS),
        "canonicalize": canonicalize,
    }
    return frame, vocabulary, manifest
