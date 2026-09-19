"""Groq chat-completions adapter (route locked by registration).

Model `openai/gpt-oss-120b`, endpoint
`https://api.groq.com/openai/v1/chat/completions`, temperature 0, seed 0,
`reasoning_effort: low`, strict `json_schema` (never `json_object`). The full
URL string is never sent — only `page_host` in a separate field plus the
delimited extract block. No retries inside the adapter (quota failures are
sealed by the driver, never hidden); transport returns a structured result
the driver seals verbatim, including failures.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from phishnet.llm.schema import MODEL_ID, strict_response_format

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TEMPERATURE = 0
SEED = 0
REASONING_EFFORT = "low"


def _parse_retry_after(resp: requests.Response, body: dict) -> float:
    val = resp.headers.get("retry-after")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    msg = body.get("error", {}).get("message", "") if isinstance(body, dict) else ""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d\.]+)s", msg)
    if m:
        h = float(m.group(1) or 0)
        mins = float(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        return h * 3600.0 + mins * 60.0 + sec
    return 30.0


def _system_prompt(prompt_version: str = "p4-v1") -> str:
    path = Path(f"src/phishnet/llm/prompts/{prompt_version}.txt")
    if not path.exists():
        raise FileNotFoundError(f"Unknown prompt version: {prompt_version}")
    return path.read_text(encoding="utf-8")


@dataclass
class Judgment:
    ok: bool
    parsed: dict | None
    fingerprint: str | None
    usage: dict
    latency_ms: float
    status: int
    error: str = ""
    raw_content: str = ""
    retry_after: float = 0.0


def judge(
    api_key: str,
    page_host: str,
    extract_text: str,
    prompt_version: str = "p4-v1",
    timeout: int = 120,
) -> tuple[dict, Judgment]:
    """One governed call. Returns (sealed_request, judgment)."""
    request_body = {
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "reasoning_effort": REASONING_EFFORT,
        "response_format": strict_response_format(),
        "messages": [
            {"role": "system", "content": _system_prompt(prompt_version)},
            {
                "role": "user",
                "content": f"page_host: {page_host}\n{extract_text}",
            },
        ],
    }
    t0 = time.time()
    try:
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
            timeout=timeout,
        )
        latency_ms = (time.time() - t0) * 1000.0
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {"_raw_text": resp.text[:4000]}
    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000.0
        return request_body, Judgment(
            ok=False,
            parsed=None,
            fingerprint=None,
            usage={},
            latency_ms=latency_ms,
            status=-1,
            error=f"{type(exc).__name__}: {exc}",
        )

    fingerprint = body.get("system_fingerprint") if isinstance(body, dict) else None
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    content: str | None = None
    if isinstance(body, dict):
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
    if status == 200 and isinstance(content, str):
        try:
            parsed = json.loads(content)
            ok = isinstance(parsed, dict)
            error = "" if ok else "parsed content is not an object"
        except json.JSONDecodeError as exc:
            parsed, ok, error = None, False, f"content is not JSON: {exc}"
    else:
        parsed, ok = None, False
        if isinstance(body, dict) and "error" in body:
            error = str(body["error"].get("message", f"http={status}"))
        else:
            error = f"http={status}" if not isinstance(content, str) else ""
    retry_after = 0.0
    if status == 429:
        retry_after = _parse_retry_after(resp, body) + 2.0
    return request_body, Judgment(
        ok=ok,
        parsed=parsed,
        fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        usage=usage if isinstance(usage, dict) else {},
        latency_ms=latency_ms,
        status=status,
        error=error,
        raw_content=content if isinstance(content, str) else "",
        retry_after=retry_after,
    )
