"""Tests for the fetch-side completion journal — offline only.

cmd_fetch persists every completed domain to a sidecar journal
(<cache>.journal.jsonl) before moving on, and replays it on resume, so a
killed run loses at most the single in-flight append instead of a whole
stratum. These tests prove:
* append + replay round-trips entries into the cache domain list;
* a torn trailing line (kill mid-append) is dropped, not fatal;
* entries already in the cache (kill between full save and compaction)
  are not duplicated;
* replay of an absent journal is a no-op returning 0.
"""

from __future__ import annotations

from pathlib import Path

import build_cc_benign as B


def _entry(domain: str, stratum: str = "s2_11_100") -> dict:
    return {
        "domain": domain,
        "stratum": stratum,
        "rank": 11,
        "index": "CC-MAIN-2026-34",
        "mechanism": "columnar",
        "note": "ok",
        "n_records": 3,
        "attempts": 1,
        "records": [],
    }


def test_journal_roundtrip(tmp_path: Path) -> None:
    journal = tmp_path / "cc-columnar-CC-MAIN-2026-34.journal.jsonl"
    B.journal_append(journal, _entry("example.com"))
    B.journal_append(journal, _entry("example.org"))
    domains: list[dict] = []
    assert B.journal_replay(journal, domains) == 2
    assert [d["domain"] for d in domains] == ["example.com", "example.org"]
    # Complete records survive the trip (rank stamped before journaling).
    assert domains[0]["rank"] == 11


def test_journal_drops_torn_tail(tmp_path: Path) -> None:
    journal = tmp_path / "c.journal.jsonl"
    B.journal_append(journal, _entry("example.com"))
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write('{"domain": "torn-examp')  # kill mid-append
    domains: list[dict] = []
    assert B.journal_replay(journal, domains) == 1
    assert [d["domain"] for d in domains] == ["example.com"]


def test_journal_no_duplicates_on_replay(tmp_path: Path) -> None:
    journal = tmp_path / "c.journal.jsonl"
    B.journal_append(journal, _entry("example.com"))
    domains: list[dict] = [_entry("example.com")]  # already fully saved
    assert B.journal_replay(journal, domains) == 0
    assert len(domains) == 1


def test_journal_replay_missing_file(tmp_path: Path) -> None:
    domains: list[dict] = []
    assert B.journal_replay(tmp_path / "absent.journal.jsonl", domains) == 0
    assert domains == []
