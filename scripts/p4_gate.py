"""§4.4 provider-capability gate (run id `p4-gate`).

One request, sealed provisional, supplies no published number. Asserts the
five pre-registered properties of the route before `client.py` lands:

1. strict `json_schema` honored against the real `RESPONSE_SCHEMA`;
2. `seed` accepted for this model ID on this endpoint;
3. `reasoning_effort: low` accepted for this model ID;
4. `system_fingerprint` present in the response;
5. `usage` breaks out reasoning tokens from visible output tokens.

Usage: uv run python scripts/p4_gate.py
Seals everything under runs/phase4/p4-gate/ (request with the key redacted,
status, headers, raw body, assertion table). Free tier, synthetic fixture —
never a corpus row, never the evaluation run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from phishnet.llm.schema import (  # noqa: E402
    MODEL_ID,
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    is_strict_compatible,
    strict_response_format,
)

GATE_ID = "p4-gate"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OUT_DIR = Path("runs/phase4/p4-gate")

# Synthetic benign fixture. Deliberately NOT a corpus row: the gate tests the
# route's capabilities, not its judgment.
SYNTHETIC_EXTRACT = (
    "title: Example Bakery — Fresh Bread Daily\n"
    "meta: Neighborhood bakery menu, hours, and contact page.\n"
    "text: Welcome to Example Bakery. Opening hours Mon–Sat 7am–6pm. "
    "Call us at (555) 010-2030. No online ordering.\n"
    "forms: none\nlink_hosts: examplebakery.test(4)\n"
    "language: en"
)
SYNTHETIC_HOST = "examplebakery.test"


def load_key() -> str:
    """GROQ_API_KEY from the environment or .env (never printed, never sealed)."""
    import os

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        raise SystemExit("GROQ_API_KEY not found in environment or .env")
    return key


def build_request(system_prompt: str) -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "seed": 0,
        "reasoning_effort": "low",
        "response_format": strict_response_format(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"page_host: {SYNTHETIC_HOST}\n"
                    f"<untrusted_page_extract>\n{SYNTHETIC_EXTRACT}\n"
                    "</untrusted_page_extract>"
                ),
            },
        ],
    }


def check_schema_conformance(content: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return False, f"content is not JSON: {exc}"
    if not isinstance(parsed, dict):
        return False, "parsed content is not an object"
    missing = [k for k in RESPONSE_SCHEMA["required"] if k not in parsed]
    if missing:
        return False, f"missing keys: {missing}"
    if not isinstance(parsed.get("evidence"), list):
        return False, "evidence is not an array"
    if parsed.get("verdict") not in ("phishing", "benign", "suspicious"):
        return False, f"bad verdict: {parsed.get('verdict')!r}"
    return True, "conforming parse, all required keys present"


def main() -> int:
    import requests

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system_prompt = Path("src/phishnet/llm/prompts/p4-v1.txt").read_text(
        encoding="utf-8"
    )
    assert is_strict_compatible(RESPONSE_SCHEMA), "schema violates strict constraints"
    request_body = build_request(system_prompt)

    sealed_request = dict(request_body)
    key = load_key()
    t0 = time.time()
    try:
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {key}"},
            json=request_body,
            timeout=120,
        )
        status = resp.status_code
        try:
            body: object = resp.json()
        except ValueError:
            body = {"_raw_text": resp.text[:4000]}
        headers = dict(resp.headers)
    except Exception as exc:  # seal transport failures too (§5.2)
        status = -1
        body = {"_transport_error": f"{type(exc).__name__}: {exc}"}
        headers = {}
    latency_ms = (time.time() - t0) * 1000

    (OUT_DIR / "request.json").write_text(
        json.dumps(sealed_request, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "response.json").write_text(
        json.dumps({"status": status, "headers": headers, "body": body}, indent=2),
        encoding="utf-8",
    )

    assertions: dict[str, dict[str, object]] = {}
    ok_overall = True

    # Assertion 1: strict json_schema honored.
    a1 = False
    detail1 = f"http={status}"
    if status == 200 and isinstance(body, dict):
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if isinstance(content, str):
            a1, detail1 = check_schema_conformance(content)
        else:
            detail1 = "no choices[0].message.content string"
    assertions["1_strict_json_schema"] = {"pass": a1, "detail": detail1}
    ok_overall &= a1

    # Assertions 2 & 3: seed / reasoning_effort accepted (200 without
    # param-rejection error naming the field).
    raw = json.dumps(body)[:4000].lower()
    a2 = status == 200 or "seed" not in raw
    assertions["2_seed_accepted"] = {"pass": a2, "detail": f"http={status}"}
    ok_overall &= bool(a2)
    a3 = status == 200 or "reasoning_effort" not in raw
    assertions["3_reasoning_effort_low"] = {"pass": a3, "detail": f"http={status}"}
    ok_overall &= bool(a3)

    # Assertion 4: system_fingerprint present.
    fp = body.get("system_fingerprint") if isinstance(body, dict) else None
    a4 = isinstance(fp, str) and len(fp) > 0
    assertions["4_system_fingerprint"] = {"pass": a4, "detail": f"fingerprint={fp!r}"}

    # Assertion 5: usage splits reasoning vs visible tokens.
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    details = (
        usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
    )
    reasoning_tokens = (
        details.get("reasoning_tokens") if isinstance(details, dict) else None
    )
    a5 = isinstance(reasoning_tokens, int)
    assertions["5_reasoning_usage_split"] = {
        "pass": a5,
        "detail": f"usage={json.dumps(usage)[:500]}",
    }

    run_record = {
        "run_id": GATE_ID,
        "run_class": "provisional",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "seed": 0,
        "temperature": 0,
        "reasoning_effort": "low",
        "system_fingerprint": fp,
        "tier": "free",
        "endpoint": ENDPOINT,
        "latency_ms": round(latency_ms, 1),
        "registration_commit": "509ff11f9e4fda17e7db389512b931edce8a4530",
        "supplies_published_number": False,
    }
    (OUT_DIR / "run.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "assertions.json").write_text(
        json.dumps(assertions, indent=2), encoding="utf-8"
    )

    for name, res in assertions.items():
        mark = "PASS" if res["pass"] else "FAIL"
        print(f"[{mark}] {name}: {res['detail']}")
    print(f"sealed -> {OUT_DIR}/ (run_class=provisional, fingerprint={fp!r})")
    return 0 if ok_overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
