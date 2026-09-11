"""Feature-extraction tests for ``phishnet.features``.

Covers ``comprehensive_phishing_features`` return structure and behavior
across representative URL shapes, plus a security regression test proving
userinfo (``user@``) is never attributed as the hostname.
"""

from urllib.parse import urlparse

from phishnet import features as features_pkg
from phishnet.features import comprehensive_phishing_features, extraction


def test_public_api_exports_extractor() -> None:
    assert callable(comprehensive_phishing_features)
    assert extraction.comprehensive_phishing_features is comprehensive_phishing_features
    assert "comprehensive_phishing_features" in features_pkg.__all__


def test_return_structure_is_numeric_feature_dict() -> None:
    feats = comprehensive_phishing_features("https://www.example.com/path")
    assert isinstance(feats, dict)
    assert feats  # non-empty
    assert isinstance(feats["tld"], str)
    for key, value in feats.items():
        if key == "tld":
            continue
        assert isinstance(value, (int, float)), key


def test_normal_https_url() -> None:
    url = "https://www.example.com/path"
    feats = comprehensive_phishing_features(url)
    assert feats["url_length"] == len(url)
    assert feats["is_https"] == 1
    assert feats["tld"] == "com"
    assert feats["path_length"] == len("/path")
    assert feats["subdomain_count"] == 1  # 'www'
    assert feats["has_at_symbol"] == 0
    assert feats["has_ip_address"] == 0


def test_http_url() -> None:
    feats = comprehensive_phishing_features("http://example.com")
    assert feats["is_https"] == 0
    assert feats["tld"] == "com"
    assert feats["query_length"] == 0
    assert feats["query_param_count"] == 0
    assert feats["has_suspicious_path"] == 0


def test_missing_scheme_defaults_to_http() -> None:
    feats = comprehensive_phishing_features("example.com/path")
    assert feats["is_https"] == 0
    assert feats["domain_length"] == len("example.com")
    assert feats["path_length"] == len("/path")
    assert feats["tld"] == "com"


def test_url_with_userinfo_flags_at_but_parses_host() -> None:
    feats = comprehensive_phishing_features("https://user@example.com/")
    assert feats["count_ats"] == 1
    assert feats["has_at_symbol"] == 1
    assert feats["is_https"] == 1
    assert feats["tld"] == "com"


def test_ipv4_host() -> None:
    feats = comprehensive_phishing_features("http://192.168.1.1/admin")
    assert feats["has_ip_address"] == 1
    assert feats["has_ip_pattern"] == 1
    assert feats["has_port"] == 0
    assert feats["has_suspicious_path"] == 1  # '/admin'
    assert feats["tld"] == ""  # no DNS suffix for a literal IP


def test_ipv6_host_does_not_crash() -> None:
    url = "http://[::1]/"
    feats = comprehensive_phishing_features(url)
    assert feats["url_length"] == len(url)
    assert feats["has_ip_address"] == 0  # implementation only matches IPv4
    assert feats["has_ip_pattern"] == 0
    # Naive ':' check also trips on the bracketed IPv6 literal colons.
    assert feats["has_port"] == 1


def test_punycode_idn_flagged() -> None:
    feats = comprehensive_phishing_features("http://xn--mnchen-3ya.de/")
    assert feats["has_suspicious_chars"] == 1  # 'xn--' prefix
    assert feats["tld"] == "de"


def test_unicode_domain() -> None:
    url = "https://münchen.de/"
    feats = comprehensive_phishing_features(url)
    assert feats["url_length"] == len(url)
    assert feats["is_https"] == 1
    assert feats["tld"] == "de"
    assert feats["has_suspicious_chars"] == 0  # no 'xn--'/percent encoding


def test_url_with_port() -> None:
    feats = comprehensive_phishing_features("https://example.com:8443/x")
    assert feats["has_port"] == 1
    assert feats["domain_length"] == len("example.com:8443")
    assert feats["is_https"] == 1
    assert feats["path_length"] == len("/x")


def test_query_string_and_fragment() -> None:
    feats = comprehensive_phishing_features("https://example.com/search?q=a&b=2#frag")
    assert feats["query_length"] == len("q=a&b=2")
    assert feats["query_param_count"] == 2
    assert feats["fragment_length"] == len("frag")
    assert feats["path_length"] == len("/search")


def test_at_userinfo_is_not_the_hostname() -> None:
    """Security regression: userinfo must not be attributed as the host.

    In ``https://google.com@evil.example/login`` everything before ``@`` is
    credentials (RFC 3986 section 3.2.1); the real hostname is
    ``evil.example``. The extractor must reflect that: ``google.com`` must
    not contribute brand/host signals.
    """
    url = "https://google.com@evil.example/login"
    assert urlparse(url).hostname == "evil.example"  # ground truth

    feats = comprehensive_phishing_features(url)
    control = comprehensive_phishing_features("https://evil.example/login")

    # The '@' itself is still flagged as a threat indicator ...
    assert feats["count_ats"] == 1
    assert feats["has_at_symbol"] == 1
    # ... and the suspicious path is still detected.
    assert feats["has_suspicious_path"] == 1

    # ... but host-derived signals match the clean control URL exactly,
    # proving the userinfo portion is inert for hostname attribution.
    assert feats["domain_length"] == control["domain_length"] == len("evil.example")
    assert feats["tld"] == control["tld"]
    assert feats["domain_token_count"] == control["domain_token_count"]

    # In particular, attacker-controlled 'google.com' must not register as
    # a trusted-brand presence or a zero-distance brand match.
    assert feats["contains_google"] == 0
    assert feats["google_min_distance"] > 0
