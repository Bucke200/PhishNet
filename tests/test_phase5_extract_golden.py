"""Phase 5 extractor output-identity test (prereg §1, criterion 2).

The fixture was generated from `phase-4-close` code
(`docs/phase5-preregistration.md` §1) and runs against CURRENT code: any
future change that alters extractor output fails here instead of passing
unnoticed. Includes a `rel`-attribute probe so the `30a1d76a` rel-branch fix
("unreachable" via bs4 parsing) is exercised, not argued.
"""

import json
from pathlib import Path

from phishnet.snapshot.extract import canonical_extract, extract_hash

FIXTURE = Path("tests/fixtures/p5-extract-golden.json")


def _probes() -> list[dict]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return loaded


def test_golden_probes_reproduce_phase4_close_output() -> None:
    probes = _probes()
    assert len(probes) == 14
    for probe in probes:
        got = canonical_extract(probe["html"], probe["page_url"])
        assert got == probe["expected_extract"], probe["probe_id"]


def test_golden_hashes_stable() -> None:
    for probe in _probes():
        recomputed = extract_hash(canonical_extract(probe["html"], probe["page_url"]))
        pinned = extract_hash(probe["expected_extract"])
        assert recomputed == pinned, probe["probe_id"]
