"""Phase 5 manifest tests (commit-2 harness checks; prereg criteria 6, 11, 16).

- Schema, group-split integrity and family coverage over the committed manifest.
- Rebuild reproducibility: the HTML stays gitignored and only hashes commit,
  so this test rebuilds everything via p5_build_pages.build() and asserts
  every sha256_raw_html / sha256_canonical_extract matches the manifest. A
  dependency bump that changes escaping or parsing fails here instead of
  silently orphaning the manifest.
- Draw chain without committing feed data: the committed draw fixture pins
  the 60 drawn rows plus the test.csv hash it came from
  (manifest -> fixture -> test.csv -> hashes-p3.json). A local-only test
  re-runs the seeded draw from the full test.csv and asserts byte-for-byte
  fixture reproduction; it skips in CI (stated reason below) where the
  fixture IS the draw.
"""

import hashlib
import json
from pathlib import Path

import pytest

import p5_build_pages as B5

MANIFEST = Path("reports/adversarial-manifest-p5.json")


def _manifest() -> list[dict]:
    loaded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return loaded


def test_frame_fixture_matches_pin() -> None:
    fixture = json.loads(B5.URL_DRAW_FIXTURE.read_text(encoding="utf-8"))
    pins = json.loads(Path("repro/hashes-p3.json").read_text(encoding="utf-8"))
    assert fixture["test_csv_sha256"] == pins["test.csv"]
    assert fixture["seed"] == B5.SEED_URL_DRAW
    assert len(fixture["rows"]) == 60


@pytest.mark.skipif(
    not Path(B5.TEST_CSV).exists(),
    reason="needs gitignored data/splits-p3/test.csv (redistributed feed "
    "data, deliberately uncommitted); CI covers the chain through the "
    "committed draw fixture instead",
)
def test_draw_reproduces_fixture() -> None:
    digest = hashlib.sha256(Path(B5.TEST_CSV).read_bytes()).hexdigest()
    assert digest == B5.load_url_draw_fixture()["test_csv_sha256"]
    redrawn = B5._draw_live(B5.expand_bases())
    pinned = json.loads(B5.URL_DRAW_FIXTURE.read_text(encoding="utf-8"))
    assert json.dumps(redrawn, sort_keys=True) == json.dumps(pinned, sort_keys=True)


def test_manifest_schema() -> None:
    required = {
        "page_id",
        "base_id",
        "kind",
        "direction",
        "template",
        "brand",
        "vector",
        "payload_id",
        "payload_family",
        "aware_rewrite_type",
        "aware_attempts",
        "aware_discards",
        "authoring_method",
        "aware_quality_rejected",
        "url",
        "tier1_score",
        "sha256_raw_html",
        "sha256_canonical_extract",
        "reached",
        "detector_hit",
        "split",
    }
    for row in _manifest():
        assert required <= set(row), row.get("page_id")
        assert row["kind"] in ("clean", "injected")
        assert row["split"] in ("dev", "held_out")
        assert row["url"] and row["tier1_score"] is not None


def test_split_integrity() -> None:
    arms: dict[str, set[str]] = {}
    for row in _manifest():
        arms.setdefault(row["base_id"], set()).add(row["split"])
    assert all(len(v) == 1 for v in arms.values()), [
        b for b, v in arms.items() if len(v) > 1
    ]
    dev = sum(1 for v in arms.values() if v == {"dev"})
    held = sum(1 for v in arms.values() if v == {"held_out"})
    assert (dev, held) == (24, 36), (dev, held)


def test_family_coverage_both_arms() -> None:
    from phishnet.adversarial.payloads import ORDINARY_EVASION

    want = {p["family"] for p in ORDINARY_EVASION}
    for arm in ("dev", "held_out"):
        got: set[str] = set()
        for row in _manifest():
            if (
                row["split"] == arm
                and row["direction"] == "evasion"
                and row["payload_family"] == "ordinary"
            ):
                got.update(
                    p["family"]
                    for p in ORDINARY_EVASION
                    if p["id"] == row["payload_id"]
                )
        assert want <= got, (arm, sorted(got))


def test_rebuild_reproduces_manifest_hashes() -> None:
    aware: list[dict] = []
    if B5.AWARE_DRAFTS.exists():
        aware = json.loads(B5.AWARE_DRAFTS.read_text(encoding="utf-8"))
    committed = {r["page_id"]: r for r in _manifest()}
    records, _ = B5.build(aware)
    assert {r.page_id for r in records} == set(committed)
    for r in records:
        pinned = committed[r.page_id]
        assert r.sha256_raw_html == pinned["sha256_raw_html"], r.page_id
        assert r.sha256_canonical_extract == pinned["sha256_canonical_extract"], (
            r.page_id
        )


@pytest.mark.skipif(
    not B5.AWARE_LOG.exists(),
    reason="aware log lands in commit 2 with the first kept drafts",
)
def test_aware_log_matches_manifest() -> None:
    log = json.loads(B5.AWARE_LOG.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in log}
    assert len(by_id) == len(log), "duplicate candidate ids in aware log"
    kept_ids = {e["id"] for e in log if e["disposition"] == "kept"}
    # Every aware page traces to a kept log entry; audit runs stay out.
    for row in _manifest():
        if row.get("payload_family") != "aware":
            continue
        assert row["payload_id"] in kept_ids, row["page_id"]
        entry = by_id[row["payload_id"]]
        assert row["aware_rewrite_type"] == entry["rewrite_type"], row["page_id"]
        assert row["authoring_method"] == entry["authoring_method"], row["page_id"]
    # Stamped aggregates equal the log-derived type totals (both views).
    tested = [e for e in log if e["disposition"] in ("kept", "detector_caught")]
    for row in _manifest():
        if row.get("payload_family") != "aware":
            continue
        t = row["aware_rewrite_type"]
        assert row["aware_attempts"] == sum(
            1 for e in tested if e["rewrite_type"] == t
        ), row["page_id"]
        assert row["aware_discards"] == sum(
            1 for e in tested if e["rewrite_type"] == t and e["disposition"] != "kept"
        ), row["page_id"]
        assert row["aware_quality_rejected"] == sum(
            1
            for e in log
            if e["rewrite_type"] == t and e["disposition"] == "quality_rejected"
        ), row["page_id"]
