"""Frozen Phase 4 response schema (`p4-v1`).

Single source of truth for the registration (§5.1): one provider-neutral dict
here, with the chat-completions `response_format` shape built by
`strict_response_format()`. The schema is deliberately small and factual — page
observations only, never a URL-guessing verdict — and every field is required
with `additionalProperties: false` so Groq strict mode (`strict: true`,
constrained decoding) accepts it verbatim. Narrow by amendment, never loosen
to `json_object`.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "openai/gpt-oss-120b"
PROMPT_VERSION = "p4-v1"
RESPONSE_SCHEMA_NAME = "p4_page_judgment"

CREDENTIAL_TYPES = ["password", "card", "otp", "email-login", "other"]
IDENTITY_VALUES = ["match", "mismatch", "unrelated", "unknown"]
VERDICT_VALUES = ["phishing", "benign", "suspicious"]

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "asks_for_credentials": {"type": "boolean"},
        "credential_types": {
            "type": "array",
            "items": {"type": "string", "enum": CREDENTIAL_TYPES},
        },
        "imitated_brand": {"type": ["string", "null"]},
        "brand_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "urgency_pressure": {"type": "boolean"},
        "urgency_score": {"type": "number", "minimum": 0, "maximum": 1},
        "identity_domain_match": {"type": "string", "enum": IDENTITY_VALUES},
        "verdict": {"type": "string", "enum": VERDICT_VALUES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": [
        "asks_for_credentials",
        "credential_types",
        "imitated_brand",
        "brand_confidence",
        "urgency_pressure",
        "urgency_score",
        "identity_domain_match",
        "verdict",
        "confidence",
        "evidence",
    ],
    "additionalProperties": False,
}


def strict_response_format() -> dict[str, Any]:
    """Chat-completions `response_format` for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": RESPONSE_SCHEMA_NAME,
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        },
    }


def is_strict_compatible(schema: dict[str, Any]) -> bool:
    """Check the documented strict-mode constraints (all required, no extras)."""
    if schema.get("type") != "object":
        return False
    props = schema.get("properties", {})
    required = schema.get("required", [])
    return (
        isinstance(props, dict)
        and set(required) == set(props)
        and schema.get("additionalProperties") is False
    )
