"""Phase 5 prompt integrity and freeze tests (§5.2, phase5-H).

- Pin SHA256 hashes of baseline p4-v1 and hardened freeze p5-h1.
- Verify phase5-H invariant: edits between p4-v1 and p5-h1 modify ONLY injection
  defense rules (delimiter integrity and instruction hierarchy), leaving clean
  evaluation rules and output requirements identical.
"""

import hashlib
from pathlib import Path

from phishnet.llm.client import _system_prompt

P4_V1_SHA256 = "f37d30df14193a1d8705b44d356fdb8645f68fc3e988b9074ad6eba2bf0a749e"
P5_H1_SHA256 = "a3c9d88c5e3ffe0605bbfb1afb2c16686675d9102ce8810606a3b473583a893b"


def test_baseline_prompt_hash_pin() -> None:
    content = _system_prompt("p4-v1")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert digest == P4_V1_SHA256


def test_hardened_freeze_prompt_hash_pin() -> None:
    content = _system_prompt("p5-h1")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert digest == P5_H1_SHA256


def test_phase5_h_injection_only_diff() -> None:
    p4_lines = (
        Path("src/phishnet/llm/prompts/p4-v1.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    h1_lines = (
        Path("src/phishnet/llm/prompts/p5-h1.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    # Header and preamble identical
    assert p4_lines[0] == h1_lines[0]
    assert p4_lines[1] == h1_lines[1]
    assert p4_lines[2] == h1_lines[2]
    # Rule 1 (judge page content, page_host) identical
    assert p4_lines[3] == h1_lines[3]

    # Shared trailing rules identical:
    # verdict, imitated_brand, identity_domain_match, evidence, confidence,
    # output schema
    p4_suffix = p4_lines[5:]
    h1_suffix = h1_lines[6:]
    assert p4_suffix == h1_suffix
