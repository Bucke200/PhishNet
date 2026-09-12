"""Tests for the Phase 2 soft-voting candidate.

The candidate must stay a pure combination-rule change: identical artifacts,
preprocessing, and data as the frozen hard-vote baseline, with continuous
member-probability scores. Feature extraction is local; no network is used.
"""

from __future__ import annotations

import numpy as np

import predictors
from eval import load_predictor


def _synthetic_urls(n: int = 150) -> list[str]:
    hosts = ["example.com", "shop.example.org", "login-secure.net", "bank-info.io"]
    paths = ["/", "/about", "/login/verify", "/a/b/c?x=1&y=2", "/news/2024/1/story"]
    return [
        f"https://{hosts[i % len(hosts)]}{paths[(i // len(hosts)) % len(paths)]}?i={i}"
        for i in range(n)
    ]


def test_candidate_loads_same_artifacts_and_preprocessing() -> None:
    legacy = predictors.LegacyEnsemble()
    cand = predictors.SoftVoteEnsemble()

    assert cand.name != legacy.name
    assert cand.mode == "soft_vote"
    assert cand.columns == legacy.columns
    urls = _synthetic_urls(20)
    np.testing.assert_allclose(cand._features(urls), legacy._features(urls))


def test_scores_equal_mean_member_probability() -> None:
    cand = predictors.SoftVoteEnsemble()
    urls = _synthetic_urls(20)
    X = cand._features(urls)
    expected = np.column_stack(
        [e.predict_proba(X)[:, 1] for e in cand.model.estimators_]
    ).mean(axis=1)

    np.testing.assert_allclose(cand.score(urls), expected, rtol=1e-12)


def test_scores_are_continuous_and_bounded() -> None:
    cand = predictors.SoftVoteEnsemble()
    urls = _synthetic_urls(150)
    scores = cand.score(urls)

    assert len(scores) == 150
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(np.isfinite(scores))
    assert len(set(scores)) > 5  # hard votes cap at 5 levels
    assert cand.score(urls) == scores  # deterministic


def test_missing_member_proba_is_loud() -> None:
    class _NoProba:
        pass

    try:
        predictors._require_member_proba([_NoProba()])
    except TypeError as e:
        assert "predict_proba" in str(e)
    else:
        raise AssertionError("expected TypeError")


def test_harness_loads_candidate_by_spec() -> None:
    pred = load_predictor("predictors:SoftVoteEnsemble")

    assert pred.name == "soft_vote(models-v1)"
    assert len(pred.score(["https://example.com/"])) == 1
