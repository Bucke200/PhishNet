"""Phase 6 serving bench: writes `reports/phase6.json`.

Registered measurements:
- C1 identity: serving fast path vs the Phase 3 headline scorer
  (`snapshot.tier1.score_band`) over the full calib and test bands.
- C2 latency: Tier-1 single-URL p50 in the Phase 3 serving shape (in-process,
  n=300, seed 0, 20 warmups) plus the §0 attribution paths.
- C2 HTTP end-to-end p50/p90, either against `--http-base` (the container) or
  a local uvicorn thread.
- §0 probe 3 re-run: shortener attribution (descriptive; C5 makes no FPR
  claim).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests
import uvicorn

from phishnet.adversarial.lexical import COVERED_SHORTENERS, shortener_wrap
from phishnet.enrichment.features import HOSTED_COLUMN
from phishnet.features.extraction import (
    canonicalize_scheme,
    comprehensive_phishing_features,
    featurise_frame,
)
from phishnet.serving import Tier1Servable
from phishnet.serving.app import create_app
from phishnet.snapshot.tier1 import score_band

CALIB = Path("data/splits-p3/calib.csv")
TEST = Path("data/splits-p3/test.csv")
OUT = Path("reports/phase6.json")
SEED = 0
N_SAMPLE = 300
WARMUP = 20
N_ATTRIB = 500


def _band_urls(path: Path) -> list[str]:
    return pd.read_csv(path, usecols=["url"])["url"].astype(str).tolist()


def _percentiles(times_ms: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(times_ms.size),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }


def tier1_identity(svc: Tier1Servable) -> dict[str, object]:
    result: dict[str, object] = {}
    for path, name in ((CALIB, "calib"), (TEST, "test")):
        _, headline = score_band(str(path))
        urls = _band_urls(path)
        served = np.array([svc.score_one(u) for u in urls], dtype=float)
        result[name] = {
            "n": len(urls),
            "max_abs_diff": float(np.max(np.abs(served - headline))),
        }
    return result


def latency_in_process(svc: Tier1Servable) -> dict[str, object]:
    urls = _band_urls(TEST)
    rng = np.random.default_rng(SEED)
    sample = list(rng.choice(urls, N_SAMPLE, replace=False))
    for u in sample[:WARMUP]:
        svc.score_one(u)
    times = []
    for u in sample:
        t0 = time.perf_counter()
        svc.score_one(u)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(np.asarray(times))


def path_attribution(svc: Tier1Servable) -> dict[str, object]:
    urls = _band_urls(CALIB)
    rng = np.random.default_rng(SEED)
    sample = list(rng.choice(urls, N_ATTRIB, replace=False))
    frozen = [c for c in svc.columns if c != HOSTED_COLUMN]
    booster = svc._booster  # noqa: SLF001 - bench instrument

    def time_extract(u: str) -> float:
        t0 = time.perf_counter()
        comprehensive_phishing_features(canonicalize_scheme(u))
        return (time.perf_counter() - t0) * 1000.0

    def time_frame(u: str) -> float:
        t0 = time.perf_counter()
        featurise_frame([u], frozen, canonicalize=True)
        return (time.perf_counter() - t0) * 1000.0

    def time_fast(u: str) -> float:
        t0 = time.perf_counter()
        booster.predict(svc.row(u))
        return (time.perf_counter() - t0) * 1000.0

    extractor: list[float] = []
    frame_path: list[float] = []
    fast_path: list[float] = []
    for i, u in enumerate(sample):
        if i < WARMUP:
            time_extract(u)
            time_frame(u)
            time_fast(u)
            continue
        extractor.append(time_extract(u))
        frame_path.append(time_frame(u))
        fast_path.append(time_fast(u))
    return {
        "extractor_only": _percentiles(np.asarray(extractor)),
        "featurise_frame_n1": _percentiles(np.asarray(frame_path)),
        "fast_row_booster": _percentiles(np.asarray(fast_path)),
    }


def shortener_probe(svc: Tier1Servable) -> dict[str, object]:
    calib = pd.read_csv(CALIB, usecols=["url", "label"])
    benign = calib[calib["label"] == 0].copy()
    rng = random.Random("p5-benign-sample:7")
    idx = list(benign.index)
    rng.shuffle(idx)
    base = benign.loc[idx[:200]]["url"].astype(str).tolist()
    wrapped = [shortener_wrap(u, h) for h in COVERED_SHORTENERS for u in base]
    t05 = svc.thresholds["t_alert"]
    t10 = svc.thresholds["t_1pct"]
    lower = svc.thresholds["lower_edge"]
    booster = svc._booster  # noqa: SLF001 - bench instrument
    short_idx = svc.columns.index("is_shortened")

    served = np.array([svc.score_one(u) for u in wrapped])
    forced = []
    for u in wrapped:
        row = svc.row(u)
        row[0, short_idx] = 0.0
        forced.append(float(booster.predict(row)[0]))
    forced = np.array(forced)
    return {
        "n": len(wrapped),
        "as_served": {
            "alert_t05": float((served >= t05).mean()),
            "alert_t10": float((served >= t10).mean()),
            "in_band": float(((served >= lower) & (served < t05)).mean()),
        },
        "flag_forced_zero": {
            "alert_t05": float((forced >= t05).mean()),
            "alert_t10": float((forced >= t10).mean()),
            "in_band": float(((forced >= lower) & (forced < t05)).mean()),
        },
    }


def http_e2e(svc: Tier1Servable, http_base: str | None) -> dict[str, object]:
    urls = _band_urls(TEST)
    rng = np.random.default_rng(SEED)
    sample = list(rng.choice(urls, N_SAMPLE, replace=False))
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    base = http_base or "http://127.0.0.1:8123"
    if http_base is None:
        app = create_app(servable=svc)
        config = uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
    server_model_hash: str | None = None
    responses: list[tuple[str, dict]] = []
    try:
        try:
            health = requests.get(f"{base}/health", timeout=30).json()
            server_model_hash = health.get("model_hash")
        except requests.RequestException:
            pass
        for u in sample[:5]:
            requests.post(f"{base}/predict", json={"url": u}, timeout=30)
        times = []
        for u in sample:
            t0 = time.perf_counter()
            response = requests.post(f"{base}/predict", json={"url": u}, timeout=30)
            times.append((time.perf_counter() - t0) * 1000.0)
            responses.append((u, response.json()))
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
    # Score-identity through the HTTP path: compare only rows the service
    # scored as given (a shortener redirect would change the scored URL).
    diffs = []
    for u, body in responses:
        if body.get("scored_url") == u and body.get("tier1_score") is not None:
            diffs.append(abs(float(body["tier1_score"]) - svc.score_one(u)))
    out = _percentiles(np.asarray(times))
    out["mode"] = "container" if http_base else "local_uvicorn"
    out["n_score_checked"] = len(diffs)
    out["max_abs_diff"] = max(diffs) if diffs else None
    out["server_model_hash"] = server_model_hash
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-base", default=None)
    args = parser.parse_args(argv)

    svc = Tier1Servable()
    report: dict[str, object] = {
        "run_class": "registered",
        "model_hash": svc.model_hash,
        "columns_hash": svc.columns_hash,
        "thresholds": svc.thresholds,
        "thresholds_source": svc.thresholds_source,
        "tier1_identity": tier1_identity(svc),
        "latency_in_process": latency_in_process(svc),
        "path_attribution": path_attribution(svc),
        "http_e2e": http_e2e(svc, args.http_base),
        "shortener_attribution": shortener_probe(svc),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
