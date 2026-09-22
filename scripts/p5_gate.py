"""Phase 5 gate: confirm model, schema and cache key on dev pages only.

Five clean dev bases (sorted-first, deterministic) judged under frozen
`p4-v1`. NO held-out page is sent here — not at the gate, not ever before
the hardened prompt freezes. Every call goes through `llm/cache.py` with run
id `p5-gate` and repeat index 0; a sealed record lands per call (success or
logged failure). Gate cap is 5 sealed SUCCESSES (`phase5-E`); attempts are
bounded (15) and failures never count.

Phase-4 gate assertions, adapted: strict `json_schema` honored (parsed object
carries exactly the registered required keys), request model ID matches,
seed/temperature/effort as registered, fingerprints recorded as a
distribution (`phase4-C` posture — rotation expected, never a single-value
predicate). Cache accounting is RECORDED, not required (`phase5-G`: this
route omits `prompt_tokens_details` on cold and warm-prefix calls alike);
the reasoning split (`completion_tokens_details.reasoning_tokens`) IS
required, since cost lines depend on it. Tokens/call (prompt, completion,
reasoning) derive the pacing interval from free-tier TPM (30 RPM / 8,000 TPM
per model, re-verified against the live rate-limits page 2026-09-19). The
measured pacing is committed here (`pacing.json`) before dev starts.

Gate calls are spaced by a conservative fixed bootstrap interval (the derived
pacing does not exist yet when the first call goes out). No resume: a failed
gate re-runs from scratch under a new run id (or after clearing the run dir);
mixing seals across attempts would muddy the token means the pacing derives
from.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, ".")

from phishnet.llm import cache as Q  # noqa: E402
from phishnet.llm.client import judge  # noqa: E402
from phishnet.llm.schema import MODEL_ID, RESPONSE_SCHEMA  # noqa: E402
from phishnet.snapshot.extract import (  # noqa: E402
    canonical_extract,
    extract_hash,
    to_model_text,
)

RUN_ID = "p5-gate"
PROMPT_VERSION = "p4-v1"
N_SUCCESS = 5
MAX_ATTEMPTS = 15
GATE_SPACING_S = 20.0
FREE_RPM = 30
FREE_TPM = 8000
TPM_MARGIN = 1.25

MANIFEST = Path("reports/adversarial-manifest-p5.json")
BODIES = Path("data/adversarial-p5/clean")
RUN_DIR = Path("runs/phase5/p5-gate")
CACHE_DIR = Path("runs/phase5/cache")


def load_key() -> str:
    import os

    key = os.environ.get("GROQ_API_KEY", "")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("GROQ_API_KEY="):
                key = stripped.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise SystemExit("GROQ_API_KEY not found")
    return key


def dev_clean_bases(n: int = N_SUCCESS) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bases = sorted(
        {r["base_id"] for r in manifest if r["kind"] == "clean" and r["split"] == "dev"}
    )
    assert len(bases) >= n, (len(bases), n)
    by_base = {}
    for r in manifest:
        if r["kind"] == "clean" and r["base_id"] in bases[:n]:
            by_base[r["base_id"]] = r
    assert len(by_base) == n, len(by_base)
    return [by_base[b] for b in sorted(by_base)]


def main() -> int:
    key = load_key()
    targets = dev_clean_bases()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sealed: list[dict] = []
    successes = 0
    attempts = 0
    fingerprints: list[str | None] = []
    cache_accounting_noted = False
    for row in targets:
        if successes >= N_SUCCESS:
            break
        body_path = BODIES / f"{row['page_id']}.html"
        raw = body_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256_raw_html"], (
            f"body drift for {row['page_id']}: judging pinned bytes only"
        )
        html = raw.decode("utf-8")
        url = str(row["url"])
        page_host = (urllib.parse.urlsplit(url).hostname or "").lower()
        extract = canonical_extract(html, url)
        assert extract_hash(extract) == row["sha256_canonical_extract"], row["page_id"]
        snap_hash = extract_hash(extract)
        ckey = Q.cache_key(snap_hash, PROMPT_VERSION, MODEL_ID, RUN_ID, 0)
        prior = Q.lookup(CACHE_DIR, ckey)
        assert prior is None, f"stale cache entry for fresh run id: {ckey}"
        record: dict | None = None
        while record is None and attempts < MAX_ATTEMPTS:
            attempts += 1
            req, judgment = judge(key, page_host, to_model_text(extract))
            usage = judgment.usage if isinstance(judgment.usage, dict) else {}
            if judgment.ok and judgment.parsed is not None:
                assert set(judgment.parsed) == set(RESPONSE_SCHEMA["required"]), (
                    "strict schema not honored",
                    sorted(judgment.parsed),
                )
                assert req.get("model") == MODEL_ID, req.get("model")
                assert req.get("seed") == 0, req.get("seed")
                assert req.get("temperature") == 0, req.get("temperature")
                assert req.get("reasoning_effort") == "low", req.get("reasoning_effort")
                details = usage.get("prompt_tokens_details", {})
                comp_details = usage.get("completion_tokens_details", {})
                if not cache_accounting_noted:
                    # phase5-G: record, don't require. This route omits
                    # prompt_tokens_details cold and warm alike.
                    print(f"cache accounting: prompt_tokens_details={details or None}")
                    cache_accounting_noted = True
                assert isinstance(comp_details, dict) and (
                    "reasoning_tokens" in comp_details
                ), ("reasoning split absent", sorted(comp_details))
                record = {
                    "page_id": row["page_id"],
                    "url": url,
                    "snapshot_hash": snap_hash,
                    "cache_key": ckey,
                    "cold_cache": True,
                    "verdict": judgment.parsed.get("verdict"),
                    "parsed": judgment.parsed,
                    "system_fingerprint": judgment.fingerprint,
                    "usage": usage,
                    "latency_ms": judgment.latency_ms,
                    "status": judgment.status,
                    "error": judgment.error,
                }
                Q.store(CACHE_DIR, ckey, record)
                successes += 1
                fingerprints.append(judgment.fingerprint)
                print(
                    f"{row['page_id']}: verdict={record['verdict']} "
                    f"fp={judgment.fingerprint} "
                    f"prompt={usage.get('prompt_tokens')} "
                    f"cached={details.get('cached_tokens')} "
                    f"completion={usage.get('completion_tokens')} "
                    f"{judgment.latency_ms:.0f}ms"
                )
            else:
                sealed.append(
                    {
                        "page_id": row["page_id"],
                        "url": url,
                        "snapshot_hash": snap_hash,
                        "cache_key": ckey,
                        "cold_cache": True,
                        "verdict": None,
                        "status": judgment.status,
                        "error": judgment.error or "unparsed",
                        "usage": usage,
                        "latency_ms": judgment.latency_ms,
                    }
                )
                print(
                    f"{row['page_id']}: FAILED "
                    f"status={judgment.status} {judgment.error}"
                )
            if successes < N_SUCCESS:
                time.sleep(GATE_SPACING_S)
        if record is not None:
            sealed.append(record)

    ok = [s for s in sealed if s.get("verdict") is not None]
    assert len(ok) == N_SUCCESS, f"gate short: {len(ok)}/{N_SUCCESS} successes"
    assert cache_accounting_noted, "cache accounting never recorded"

    def mean(field: str) -> float:
        vals = [
            s["usage"].get(field, 0) for s in ok if isinstance(s.get("usage"), dict)
        ]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def mean_cached() -> float:
        vals = [
            (s["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            for s in ok
            if isinstance(s.get("usage"), dict)
        ]
        return float(sum(vals) / len(vals)) if vals else 0.0

    prompt_m, cached_m, completion_m = (
        mean("prompt_tokens"),
        mean_cached(),
        mean("completion_tokens"),
    )
    total_m = prompt_m + completion_m
    interval = max(60.0 / FREE_RPM, math.ceil(total_m * 60.0 / FREE_TPM * TPM_MARGIN))
    pacing = {
        "prompt_tokens_mean": prompt_m,
        "cached_tokens_mean": cached_m,
        "completion_tokens_mean": completion_m,
        "total_tokens_mean": total_m,
        "free_tpm": FREE_TPM,
        "free_rpm": FREE_RPM,
        "tpm_margin": TPM_MARGIN,
        "pacing_interval_s": interval,
        "tpm_source": (
            "console.groq.com/docs/rate-limits, re-verified live 2026-09-19 "
            "(30/1K/8K/200K for openai/gpt-oss-120b); re-check at report time"
        ),
        "n_calls": len(ok),
        "rate_source": "measured cold-cache gate calls",
    }
    (RUN_DIR / "pacing.json").write_text(
        json.dumps(pacing, indent=1) + "\n", encoding="utf-8"
    )
    distinct_fp = sorted({f for f in fingerprints if f})
    run = {
        "run_id": RUN_ID,
        "run_class": "provisional",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "seed": 0,
        "tier": "free",
        "n_success": len(ok),
        "n_failures": len(sealed) - len(ok),
        "n_attempts": attempts,
        "assertions": {
            "strict_schema_required_keys": sorted(RESPONSE_SCHEMA["required"]),
            "model_id_matches": True,
            "seed_temperature_effort_pinned": True,
            "cache_accounting_recorded_where_present": True,
            "reasoning_split_present": True,
            "bodies_hash_verified": True,
            "extract_hash_verified": True,
            "no_held_out_touched": True,
        },
        "fingerprints": distinct_fp,
        "n_fingerprints": len(distinct_fp),
        "identity_note": (
            "phase4-C: fingerprint recorded as a distribution, "
            "never a single-value predicate"
        ),
        "pacing_interval_s": interval,
    }
    (RUN_DIR / "run.json").write_text(
        json.dumps(run, indent=1) + "\n", encoding="utf-8"
    )
    (RUN_DIR / "calls.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in sealed), encoding="utf-8"
    )
    print(f"gate pass: {len(ok)}/{N_SUCCESS} successes, pacing {interval}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
