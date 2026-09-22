"""Frozen response schemas.

`p4-v1` (and its prompt-only hardened sibling `p5-h1`) is the Phase 4/5
schema, unchanged. `p6-v1` widens ``credential_types`` to add the four
values the provider rejected in 47 of the 50 Phase 5 schema failures
(`login`, `credentials`, `generic_form`, `session_token`) and changes
nothing else; the C4 replay re-sends the sealed failures under it.

The chat-completions `response_format` shape is built by
`strict_response_format(version)`. Each schema is deliberately small and
factual — page observations only, never a URL-guessing verdict — and every
field is required with `additionalProperties: false` so Groq strict mode
(`strict: true`, constrained decoding) accepts it verbatim. Narrow by
amendment, never loosen to `json_object`.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "openai/gpt-oss-120b"
PROMPT_VERSION = "p4-v1"
RESPONSE_SCHEMA_NAME = "p4_page_judgment"

CREDENTIAL_TYPES = ["password", "card", "otp", "email-login", "other"]
# phase6-C4 widening: values surfaced by real harvest pages that the p4-v1
# enum omitted (47/50 Phase 5 schema failures were this omission).
P6_CREDENTIAL_TYPES = [
    *CREDENTIAL_TYPES,
    "login",
    "credentials",
    "generic_form",
    "session_token",
]
IDENTITY_VALUES = ["match", "mismatch", "unrelated", "unknown"]
VERDICT_VALUES = ["phishing", "benign", "suspicious"]


def _response_schema(credential_types: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "asks_for_credentials": {"type": "boolean"},
            "credential_types": {
                "type": "array",
                "items": {"type": "string", "enum": credential_types},
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


RESPONSE_SCHEMA: dict[str, Any] = _response_schema(CREDENTIAL_TYPES)
P6_RESPONSE_SCHEMA: dict[str, Any] = _response_schema(P6_CREDENTIAL_TYPES)

# prompt_version -> (schema, schema name). p5-h1 is prompt-only over p4-v1.
SCHEMAS: dict[str, dict[str, Any]] = {
    "p4-v1": RESPONSE_SCHEMA,
    "p5-h1": RESPONSE_SCHEMA,
    "p6-v1": P6_RESPONSE_SCHEMA,
}
SCHEMA_NAMES: dict[str, str] = {
    "p4-v1": RESPONSE_SCHEMA_NAME,
    "p5-h1": RESPONSE_SCHEMA_NAME,
    "p6-v1": "p6_page_judgment",
}


def response_schema(prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    """The response schema for a prompt version."""
    try:
        return SCHEMAS[prompt_version]
    except KeyError as e:
        raise ValueError(f"unknown prompt version: {prompt_version}") from e


def strict_response_format(prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    """Chat-completions `response_format` for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": SCHEMA_NAMES[prompt_version],
            "strict": True,
            "schema": response_schema(prompt_version),
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
