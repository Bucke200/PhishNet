"""Phase 5 cache tests (prereg §8, criterion 14).

The `phase4-D` defect was a key without run id or repeat index: repeats 2–3
would have returned repeat 1's cached responses and the range would have had
zero width. These tests assert the fixed key shape and simulate two repeats
of one page through a stub judge: both must execute as sealed calls.
"""

from pathlib import Path

from phishnet.llm.cache import PHASE5_CACHE_DIR, cache_key, lookup, store


def _key(repeat_idx: int) -> str:
    return cache_key("snap", "p5-h1", "openai/gpt-oss-120b", "p5-run-1", repeat_idx)


def test_repeat_index_distinguishes_keys() -> None:
    assert _key(0) != _key(1) != _key(2)
    assert _key(0) == cache_key("snap", "p5-h1", "openai/gpt-oss-120b", "p5-run-1", 0)


def test_run_id_distinguishes_keys() -> None:
    a = cache_key("snap", "p5-h1", "openai/gpt-oss-120b", "run-a", 0)
    b = cache_key("snap", "p5-h1", "openai/gpt-oss-120b", "run-b", 0)
    assert a != b


def test_phase5_cache_dir_is_disjoint_from_phase4() -> None:
    assert PHASE5_CACHE_DIR == Path("runs/phase5/cache")
    assert PHASE5_CACHE_DIR != Path("runs/phase4/cache")


def test_two_repeats_of_one_page_produce_two_sealed_calls(tmp_path: Path) -> None:
    calls = []

    def stub_judge(repeat_idx: int) -> dict:
        calls.append(repeat_idx)
        return {"repeat_idx": repeat_idx, "verdict": "phishing"}

    sealed = 0
    for repeat_idx in (0, 1):
        key = _key(repeat_idx)
        record = lookup(tmp_path, key)
        if record is None:
            record = stub_judge(repeat_idx)
            store(tmp_path, key, record)
            sealed += 1
    assert sealed == 2
    assert calls == [0, 1]
    assert lookup(tmp_path, _key(0)) != lookup(tmp_path, _key(1))


def test_phase4_style_key_cannot_collide(tmp_path: Path) -> None:
    import hashlib

    legacy = hashlib.sha256(b"snap|p4-v1|openai/gpt-oss-120b").hexdigest()[:32]
    store(tmp_path, legacy, {"verdict": "benign"})
    assert lookup(tmp_path, _key(0)) is None
