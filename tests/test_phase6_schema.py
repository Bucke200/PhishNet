"""C4 — `p6-v1` schema widening.

`p6-v1` adds exactly `login`, `credentials`, `generic_form`, `session_token`
to the `credential_types` enum and changes nothing else. `p4-v1` and `p5-h1`
stay frozen and loadable; `p5-h1` is prompt-only over the `p4-v1` schema.
"""

from __future__ import annotations

import pytest

from phishnet.llm import schema


def _enum(s: dict) -> list[str]:
    return list(s["properties"]["credential_types"]["items"]["enum"])


def test_p6_adds_exactly_four_credential_types() -> None:
    p4 = set(_enum(schema.RESPONSE_SCHEMA))
    p6 = set(_enum(schema.P6_RESPONSE_SCHEMA))
    assert p6 - p4 == {"login", "credentials", "generic_form", "session_token"}
    assert p4 - p6 == set()


def test_p6_changes_nothing_else() -> None:
    p4 = schema.RESPONSE_SCHEMA
    p6 = schema.P6_RESPONSE_SCHEMA
    assert p4["required"] == p6["required"]
    assert p4["additionalProperties"] == p6["additionalProperties"] is False
    for key, value in p4["properties"].items():
        if key == "credential_types":
            continue
        assert p6["properties"][key] == value


def test_p4_is_unchanged_and_loadable() -> None:
    assert schema.CREDENTIAL_TYPES == [
        "password",
        "card",
        "otp",
        "email-login",
        "other",
    ]
    assert schema.response_schema("p4-v1") is schema.RESPONSE_SCHEMA


def test_p5_h1_is_prompt_only_over_p4_schema() -> None:
    assert schema.response_schema("p5-h1") is schema.RESPONSE_SCHEMA


def test_strict_response_format_names_versions() -> None:
    p4 = schema.strict_response_format("p4-v1")["json_schema"]
    p6 = schema.strict_response_format("p6-v1")["json_schema"]
    assert p4["name"] == "p4_page_judgment"
    assert p6["name"] == "p6_page_judgment"
    assert p6["strict"] is True


def test_p6_schema_is_strict_compatible() -> None:
    assert schema.is_strict_compatible(schema.P6_RESPONSE_SCHEMA)


def test_unknown_prompt_version_raises() -> None:
    with pytest.raises(ValueError):
        schema.response_schema("p9-v9")
