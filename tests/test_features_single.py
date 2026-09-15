"""Exact-match tests for the single-URL feature fast path.

``LegacyEnsemble._features`` dispatches n=1 calls to ``_features_single``,
which skips DataFrame construction. The batch path is already pinned to the
canonical extractor (see test_canonical_wiring.py); these tests pin the fast
path to the batch path, so transitively the fast path is canonical too.
"""

from __future__ import annotations

import numpy as np

import predictors

URLS = [
    "https://www.example.com/path",
    "http://192.168.1.1/admin",
    "example.com/path",
    "https://google.com@evil.example/login",
    "https://münchen.de/",
    "http://xn--mnchen-3ya.de/",
    "https://example.com:8443/x?q=a&b=2#frag",
    "br-icloud.com.br",
    "http://buzzfil.net/m/show-art/ils-etaient-loin-de-s-imaginer-que-le-hibou-allait",
    "https://repl-mess.myfreesites.net/",
]


def test_single_matches_batched_rows_exactly() -> None:
    """The load-bearing test: multi-row (pandas) vs single-row (numpy)."""
    ens = predictors.LegacyEnsemble()
    batched = ens._features(URLS)
    assert batched.shape == (len(URLS), len(ens.columns))
    for i, url in enumerate(URLS):
        np.testing.assert_array_equal(batched[[i]], ens._features_single(url))


def test_single_dispatch_matches_direct_call() -> None:
    """n=1 dispatch and the direct method are the same code path's output."""
    ens = predictors.LegacyEnsemble()
    for url in URLS:
        np.testing.assert_array_equal(ens._features([url]), ens._features_single(url))


def test_scores_identical_single_vs_batch() -> None:
    """End to end: per-URL scoring equals batched scoring for every mode.

    Only models-v1 predictors here: CC assets are git-ignored training
    outputs, unavailable in CI (same reason no existing test instantiates
    CcRetrained). The fast path itself is asset-independent — covered above
    against the shared ``_features`` implementation every subclass inherits.
    """
    for cls in (
        predictors.LegacyEnsemble,
        predictors.SoftVoteEnsemble,
    ):
        pred = cls()
        # allclose, not ==: tree predict_proba vectorizes differently across
        # batch sizes (last-ulp, ~1e-16 measured). Exactness is asserted at
        # the feature level above; estimator numerics are not bit-stable.
        np.testing.assert_allclose(
            pred.score(URLS), [pred.score([u])[0] for u in URLS], rtol=1e-12
        )
