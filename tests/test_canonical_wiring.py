"""Wiring tests proving production and training use the canonical extractor.

Canonical implementation: ``src/phishnet/features/extraction.py``.

* production -> ``phishnet.features.extraction``
  (via ``phishnet.api.preprocess_single_url_traditional``)
* training   -> ``phishnet.features.extraction``
  (via ``ml_training.preprocess_urlset.extract_features_from_df_urlset``)
* tests      -> ``phishnet.features.extraction`` (see ``test_features.py``)

These tests assert behavioral equivalence with the canonical implementation,
not merely that an import resolves. The feature extractor itself is never
mocked.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import ml_training.preprocess_urlset as preprocess_urlset
from phishnet.api import preprocess_single_url_traditional
from phishnet.features.extraction import comprehensive_phishing_features


def _canonical_numeric_frame(urls: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Replicate the shared extract-drop-coerce pipeline for given URLs."""
    rows = [comprehensive_phishing_features(url) for url in urls]
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=["tld"])
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    return frame, frame.columns.tolist()


def test_production_preprocess_matches_canonical_pipeline() -> None:
    urls = [
        "https://www.example.com/path",
        "http://192.168.1.1/admin",
        "example.com/path",
    ]
    frame, columns = _canonical_numeric_frame(urls)
    scaler = StandardScaler()
    scaler.fit(frame.values)

    url = urls[0]
    got = preprocess_single_url_traditional(url, scaler, columns)
    expected = scaler.transform(frame.values[[0]])

    assert got.shape == expected.shape == (1, len(columns))
    assert np.allclose(got, expected)


def test_production_preprocess_drops_tld_string_column() -> None:
    url = "https://www.example.com/path"
    raw = comprehensive_phishing_features(url)
    assert isinstance(raw["tld"], str)

    frame, columns = _canonical_numeric_frame([url, "http://192.168.1.1/admin"])
    assert "tld" not in columns

    scaler = StandardScaler()
    scaler.fit(frame.values)
    got = preprocess_single_url_traditional(url, scaler, columns)
    assert got.shape == (1, len(columns))


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
