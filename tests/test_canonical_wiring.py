"""Wiring tests proving production and training use the canonical extractor.

Canonical implementation: ``src/phishnet/features/extraction.py``.

* production -> ``phishnet.features.extraction``
  (via ``phishnet.serving.tier1.Tier1Servable``, the Phase 6 serving path)
* training   -> ``phishnet.features.extraction``
  (via ``ml_training.preprocess_urlset.extract_features_from_df_urlset``)
* tests      -> ``phishnet.features.extraction`` (see ``test_features.py``)

These tests assert behavioral equivalence with the canonical implementation,
not merely that an import resolves. The feature extractor itself is never
mocked.
"""

import inspect

import pandas as pd

import ml_training.preprocess_urlset as preprocess_urlset
import phishnet.serving.app as serving_app
import phishnet.serving.tier1 as serving_tier1
from phishnet.features.extraction import (
    canonicalize_scheme,
    comprehensive_phishing_features,
)


def test_training_preprocess_matches_canonical_extractor() -> None:
    urls = [
        "https://www.example.com/path",
        "http://192.168.1.1/admin",
    ]
    df = pd.DataFrame({"domain": urls, "label": [0, 1]})

    # The training module must reference the canonical function object.
    assert (
        preprocess_urlset.comprehensive_phishing_features
        is comprehensive_phishing_features
    )

    features_df = preprocess_urlset.extract_features_from_df_urlset(df)
    assert len(features_df) == len(urls)

    for i, url in enumerate(urls):
        expected = comprehensive_phishing_features(url)
        assert set(features_df.columns) == set(expected.keys())
        for key, value in expected.items():
            assert features_df.loc[i, key] == value, key


def test_serving_uses_canonical_extractor() -> None:
    """The Phase 6 serving fast path is built on the canonical extractor."""
    assert (
        serving_tier1.comprehensive_phishing_features is comprehensive_phishing_features
    )
    assert serving_tier1.canonicalize_scheme is canonicalize_scheme


def test_no_whitelist_short_circuit_in_serving() -> None:
    """Step 2 invariant: /predict scores every URL through the model.

    The whitelist (bypass list + ``whitelisted`` response key) was removed
    because the harness measured a pipeline the extension did not run.
    Re-adding either half must fail here, not in production: the constant
    by name, the response key by the exact spelling (the surviving code
    comment says "whitelist", never "whitelisted", so this does not trip
    on its own documentation).
    """
    assert not hasattr(serving_app, "WHITELISTED_DOMAINS")
    assert "whitelisted" not in inspect.getsource(serving_app.predict_one)
    assert "whitelisted" not in inspect.getsource(serving_app)
