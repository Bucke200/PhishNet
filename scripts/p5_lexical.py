"""Phase 5 lexical-evasion arm (§7 + `phase5-E`): Tier-1 recall under URL
transforms. Zero LLM calls.

Base: N=200 test-split phishing URLs, seeded shuffle (`p5-lexical-sample:7`),
first 200, no replacement. Thresholds: pinned `phase5-E` values, asserted by
recomputation (fail loudly on drift). Scoring: row-(a) in production mode (no
first_seen map — synthetic URLs have none); eval==production equality is
asserted on the base set (row (a) takes no age input, so mode cannot move
scores; verified here, not assumed). Metrics: recall per arm at both
thresholds, Wilson per cell, `paired_bootstrap_ci` on the recall difference
(`n_boot=2000`, seed 7). Pooled multi-variant arms treat judgments as the
unit — within-base correlation makes those CIs a lower bound on width
(stated, same posture as §4.3).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import eval as E  # noqa: E402
import predictors  # noqa: E402
from phishnet.adversarial.lexical import (  # noqa: E402
    COVERED_SHORTENERS,
    REDIRECT_HOSTS,
    UNCOVERED_SHORTENERS,
    homoglyph_domain,
    is_applicable,
    open_redirect_wrap,
    shortener_wrap,
    to_ascii_form,
)
from phishnet.snapshot.tier1 import (  # noqa: E402
    PINNED_RUN,
    PINNED_SNAPSHOT,
    ROW_A_ASSETS,
    ROW_A_COLUMNS_HASH,
    ROW_A_MODEL_HASH,
    score_band,
)

N_BASE = 200
SEED_SAMPLE = "p5-lexical-sample:7"
T_05 = 0.9269363298832987
T_10 = 0.8780843789420926
N_BOOT = 2000
BOOT_SEED = 7
TEST_CSV = "data/splits-p3/test.csv"
OUT_JSON = Path("reports/phase5-lexical.json")
OUT_MD = Path("reports/phase5-lexical.md")


def load_row_a_production() -> predictors.EnrichedGbm:
    """Row-(a) scorer without a first_seen map, hash-asserted like load_row_a."""
    pred = predictors.EnrichedGbm(
        assets_dir=ROW_A_ASSETS,
        snapshot=PINNED_SNAPSHOT,
        run_id=PINNED_RUN,
        first_seen_csv=None,
    )
    fp = pred.asset_fingerprint
    assert fp.get("model") == ROW_A_MODEL_HASH, "row-(a) model hash mismatch"
    assert fp.get("columns") == ROW_A_COLUMNS_HASH, "row-(a) columns hash mismatch"
    return pred


def recall_at(scores: np.ndarray, thr: float) -> float:
    return float((np.asarray(scores) >= thr).mean())


def main() -> int:
    test = pd.read_csv(TEST_CSV, usecols=["url", "label"])
    phish = test[test["label"] == 1].copy()
    rng = random.Random(SEED_SAMPLE)
    idx = list(phish.index)
    rng.shuffle(idx)
    base = phish.loc[idx[:N_BASE]].reset_index(drop=True)
    assert len(base) == N_BASE, len(base)
    urls = base["url"].astype(str).tolist()

    pred = load_row_a_production()

    # Mode-independence: eval scores (test.csv as its own map) vs production.
    from phishnet.snapshot.tier1 import load_row_a

    eval_pred = load_row_a(TEST_CSV)
    s_eval = np.asarray(E.score_all(eval_pred, urls, batch_size=512)[0], dtype=float)
    s_prod = np.asarray(E.score_all(pred, urls, batch_size=512)[0], dtype=float)
    mode_diff = float(np.max(np.abs(s_eval - s_prod)))
    assert mode_diff == 0.0, mode_diff

    # Thresholds: recompute, assert against the phase5-E pins.
    y_calib, s_calib = score_band("data/splits-p3/calib.csv")
    t05 = float(E.threshold_at_fpr(y_calib, np.asarray(s_calib), 0.005))
    t10 = float(E.threshold_at_fpr(y_calib, np.asarray(s_calib), 0.01))
    assert t05 == T_05, (t05, T_05)
    assert t10 == T_10, (t10, T_10)

    # Variants per base URL.
    uni = [homoglyph_domain(u) if is_applicable(u, "homoglyph") else None for u in urls]
    applicable = [u is not None for u in uni]
    asciis = [to_ascii_form(u) if u is not None else None for u in uni]
    shorts = {
        h: [shortener_wrap(u, h) for u in urls]
        for h in (*COVERED_SHORTENERS, *UNCOVERED_SHORTENERS)
    }
    redirs = {h: [open_redirect_wrap(u, h) for u in urls] for h in REDIRECT_HOSTS}

    # Score everything flat, then map back.
    jobs: dict[str, list[str]] = {"clean": urls}
    jobs["homoglyph_unicode"] = [
        u if u is not None else urls[i] for i, u in enumerate(uni)
    ]
    jobs["xn--"] = [a if a is not None else urls[i] for i, a in enumerate(asciis)]
    for h, vs in (*shorts.items(), *redirs.items()):
        jobs[f"short:{h}" if h in shorts else f"redir:{h}"] = vs
    flat: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for name, vs in jobs.items():
        spans[name] = (len(flat), len(flat) + len(vs))
        flat.extend(vs)
    scores = np.asarray(E.score_all(pred, flat, batch_size=512)[0], dtype=float)
    got = {name: scores[a:b] for name, (a, b) in spans.items()}

    arms: dict[str, dict] = {}

    def cell(name: str, mask: list[bool] | None = None) -> dict:
        s = got[name]
        m = np.asarray(mask if mask is not None else [True] * len(s), dtype=bool)
        ss = s[m]
        base_clean = got["clean"][m] if mask is not None else got["clean"]
        out = {}
        for thr_key, thr in (("t05", T_05), ("t10", T_10)):
            k = int((ss >= thr).sum())
            n = int(m.sum())
            lo, hi = E.wilson_interval(k, n)
            fn = lambda yy, sxx, _t=thr: float((np.asarray(sxx) >= _t).mean())  # noqa: E731
            dlo, dhi = E.paired_bootstrap_ci(
                fn, np.ones(n, dtype=int), ss, np.asarray(base_clean), N_BOOT, BOOT_SEED
            )
            out[thr_key] = {
                "recall": k / n,
                "k": k,
                "n": n,
                "wilson": [float(lo), float(hi)],
                "paired_diff_ci": [float(dlo), float(dhi)],
            }
        return out

    arms["clean"] = cell("clean")
    arms["clean"]["n_applicable"] = N_BASE
    arms["homoglyph_unicode"] = cell("homoglyph_unicode", applicable)
    arms["homoglyph_unicode"]["n_applicable"] = int(sum(applicable))
    arms["homoglyph_unicode"]["descriptive"] = True
    arms["xn--"] = cell("xn--", applicable)
    arms["xn--"]["n_applicable"] = int(sum(applicable))
    covered = np.concatenate([got[f"short:{h}"] for h in COVERED_SHORTENERS])
    uncovered = np.concatenate([got[f"short:{h}"] for h in UNCOVERED_SHORTENERS])
    pooled_redir = np.concatenate([got[f"redir:{h}"] for h in REDIRECT_HOSTS])
    clean_rep = np.tile(got["clean"], len(COVERED_SHORTENERS))
    for key, ss, cc in (
        ("short_covered", covered, clean_rep),
        ("short_uncovered", uncovered, clean_rep),
        ("redirect_pooled", pooled_redir, np.tile(got["clean"], len(REDIRECT_HOSTS))),
    ):
        row: dict = {"n_applicable": int(ss.size)}
        for thr_key, thr in (("t05", T_05), ("t10", T_10)):
            k = int((ss >= thr).sum())
            lo, hi = E.wilson_interval(k, int(ss.size))
            fn = lambda yy, sxx, _t=thr: float((np.asarray(sxx) >= _t).mean())  # noqa: E731
            dlo, dhi = E.paired_bootstrap_ci(
                fn, np.ones(int(ss.size), dtype=int), ss, cc, N_BOOT, BOOT_SEED
            )
            row[thr_key] = {
                "recall": k / int(ss.size),
                "k": k,
                "n": int(ss.size),
                "wilson": [float(lo), float(hi)],
                "paired_diff_ci": [float(dlo), float(dhi)],
            }
        arms[key] = row
    per_host = {}
    for name in [n for n in jobs if n.startswith(("short:", "redir:"))]:
        per_host[name] = cell(name)
        per_host[name]["descriptive"] = True

    report = {
        "n_base": N_BASE,
        "seed_sample": SEED_SAMPLE,
        "thresholds": {"t05": T_05, "t10": T_10, "recomputed_equal": True},
        "mode_independence_max_abs_diff": mode_diff,
        "n_homoglyph_inapplicable": N_BASE - int(sum(applicable)),
        "arms": arms,
        "per_host_descriptive": per_host,
        "base_urls": urls,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    def fmt_row(label: str, a: dict, key: str) -> str:
        c = a[key]
        lo, hi = c["wilson"]
        dlo, dhi = c["paired_diff_ci"]
        return (
            f"| {label} | {c['recall']:.4f} ({c['k']}/{c['n']}) | "
            f"[{lo:.4f}, {hi:.4f}] | [{dlo:+.4f}, {dhi:+.4f}] |"
        )

    lines = [
        "# Phase 5 lexical-evasion arm — Tier-1 recall under URL transforms",
        "",
        f"Base: {N_BASE} test-split phishing URLs (`{SEED_SAMPLE}`). Thresholds "
        f"pinned in `phase5-E` (`t05={T_05}`, `t10={T_10}`), recomputation "
        "asserted. Production-mode scoring; eval==production max abs diff "
        f"{mode_diff} (asserted 0.0). `paired_bootstrap_ci` on the recall "
        f"difference vs clean (`n_boot={N_BOOT}`, seed {BOOT_SEED}). Pooled "
        "multi-variant CIs treat judgments as the unit (lower bound on width, "
        "stated). Homoglyph-unicode is descriptive; ASCII (`xn--`) is primary. "
        f"Homoglyph not applicable: {N_BASE - int(sum(applicable))} rows.",
        "",
        "## Recall at t_0.5%",
        "",
        "| arm | recall (k/n) | Wilson 95% | paired diff vs clean |",
        "|---|---|---|---|",
        fmt_row("clean", arms["clean"], "t05"),
        fmt_row("xn-- (primary)", arms["xn--"], "t05"),
        fmt_row("homoglyph-unicode (descriptive)", arms["homoglyph_unicode"], "t05"),
        fmt_row("shortener covered (pooled)", arms["short_covered"], "t05"),
        fmt_row("shortener uncovered (pooled)", arms["short_uncovered"], "t05"),
        fmt_row("redirect pooled", arms["redirect_pooled"], "t05"),
        "",
        "## Recall at t_1.0%",
        "",
        "| arm | recall (k/n) | Wilson 95% | paired diff vs clean |",
        "|---|---|---|---|",
        fmt_row("clean", arms["clean"], "t10"),
        fmt_row("xn-- (primary)", arms["xn--"], "t10"),
        fmt_row("homoglyph-unicode (descriptive)", arms["homoglyph_unicode"], "t10"),
        fmt_row("shortener covered (pooled)", arms["short_covered"], "t10"),
        fmt_row("shortener uncovered (pooled)", arms["short_uncovered"], "t10"),
        fmt_row("redirect pooled", arms["redirect_pooled"], "t10"),
        "",
        "Per-host tables (descriptive) are in `reports/phase5-lexical.json` "
        "under `per_host_descriptive`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    for key in ("clean", "xn--", "short_covered", "short_uncovered", "redirect_pooled"):
        c05, c10 = arms[key]["t05"], arms[key]["t10"]
        print(f"{key}: t05={c05['recall']:.4f} t10={c10['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
