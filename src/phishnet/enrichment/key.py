"""Cache-key rule: tenant host for hosted platforms, registrable domain otherwise.

The rule is frozen here so batch enrichment, the snapshot store, the stub,
and the Phase-6 live provider all key identically. Hosted tenants share the
platform's registrable domain (and its age/certs), so keying them by domain
would smear one platform's history across thousands of unrelated tenants —
and, worse, hand every `*.core.windows.net` phish Microsoft's registration
date. Those rows are keyed by full host and marked not-applicable for
age/CT (the platform's records are not the tenant's).

`check_psl_splits` runs *before* this rule is relied upon: it exercises the
pinned PSL snapshot over hosted-platform hosts and records how they group,
so a snapshot change that regroups them is visible instead of silent.
"""

from __future__ import annotations

from urllib.parse import urlparse

import tldextract

# Hosts whose subdomains are unrelated tenants. Keyed by full host, marked
# na for age/CT (the platform's records are not the tenant's).
#
# CURATION SOURCE (phishing-independent, by construction): the PSL private
# section (publicsuffix.org private domains — infrastructure operators who
# self-report tenant-carrier suffixes) plus cloud vendors' published
# endpoint-suffix lists. NEVER derived from the phishing feeds: building
# this list from platforms seen in phishing would make age_na/ct_na mean
# "on a platform phishers use", and benign CC rows almost never land on
# those tenants — a curated label proxy. Extensions require a cited
# non-feed source in the same format (PSL-private or vendor-published).
HOSTED_PLATFORMS = frozenset(
    {
        "core.windows.net",
        "azurewebsites.net",
        "blogspot.com",
        "blogspot.co.uk",
        "wordpress.com",
        "weebly.com",
        "wixsite.com",
        "github.io",
        "gitlab.io",
        "herokuapp.com",
        "appspot.com",
        "cloudfront.net",
        "amazonaws.com",
        "s3.amazonaws.com",
        "firebaseapp.com",
        "web.app",
        "pages.dev",
        "netlify.app",
        "vercel.app",
        "glitch.me",
        "ngrok.io",
        "duckdns.org",
        "ddns.net",
        "bit.ly",  # shortener host itself is the tenant carrier; age/CT na
    }
)

_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=".tld_cache",
    suffix_list_urls=(),  # no network, ever — pinned snapshot only
    fallback_to_snapshot=True,
)


def registrable_domain(host: str) -> str:
    """Registrable domain (eTLD+1) for a host under the pinned snapshot."""
    e = _EXTRACTOR(host)
    return f"{e.domain}.{e.suffix}" if e.suffix else e.domain


def host_of(url: str) -> str:
    """Lowercased hostname for a URL (no port, no credentials)."""
    p = urlparse(url if "://" in url else "http://" + url)
    host = (p.hostname or "").lower().strip(".")
    return host


def cache_key(url: str) -> tuple[str, bool]:
    """Return (key, is_hosted_tenant) for a URL.

    Hosted tenants key by full host; everything else keys by registrable
    domain. IP literals and empty hosts key by themselves (never na —
    there is no platform to attribute, lookups just usually fail).
    """
    h = host_of(url)
    if not h:
        return h, False
    # Tenant-carrier match is suffix-based, not registrable-equality: under
    # the pinned snapshot e.g. login.core.windows.net groups to
    # windows.net, so an equality check against core.windows.net misses it.
    if h in HOSTED_PLATFORMS or any(h.endswith("." + p) for p in HOSTED_PLATFORMS):
        return h, True
    reg = registrable_domain(h)
    return reg or h, False


def hosted_share(urls: list[str]) -> dict[str, object]:
    """Share of URLs on hosted tenants (for the split manifest).

    Reported after campaign capping: the split groups by registrable
    domain under the pinned snapshot (which ignores the PSL private
    section by default), so all tenants of a platform count as ONE domain
    for straddler-dropping and the 5/50 campaign caps — hosted phish are
    capped hard, and the raw hosted share will not survive into the split.
    That grouping is kept on purpose (domain-disjointness is load-bearing
    for the CIs) and recorded here with the numbers, not silently.
    """
    keys = [cache_key(u) for u in urls]
    n = len(keys)
    hosted = sum(1 for _, h in keys if h)
    return {
        "n": n,
        "n_hosted": hosted,
        "hosted_share": (hosted / n) if n else 0.0,
    }


def gate_psl_snapshot(expected_sha256: str | None) -> str:
    """Fail the build when the runtime PSL snapshot differs from pinned.

    `expected_sha256=None` records-but-passes (first build of a new
    population); a provided value mismatching the bundled snapshot exits
    non-zero with no files written — a gate, not a reminder to run a
    check. Wire via `--expect-psl-sha`.
    """
    import hashlib as _h
    import sys as _s

    import tldextract as _t

    snap = (
        __import__("pathlib").Path(_t.__file__).resolve().parent / ".tld_set_snapshot"
    )
    actual = _h.sha256(snap.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        print(
            f"PSL SNAPSHOT MISMATCH: expected {expected_sha256}, "
            f"runtime {actual} — refusing to build.",
            file=_s.stderr,
        )
        raise SystemExit(1)
    return actual


def check_psl_splits(
    urls: list[str], expected_sha256: str | None = None
) -> dict[str, object]:
    """Exercise the pinned snapshot over hosted hosts; report groupings.

    Call before relying on `cache_key` for a new population: returns the
    snapshot identity plus per-URL (host, registrable_domain, key, hosted)
    rows so a regroup is diffable. Pure/offline — asserts no network by
    construction (extractor has empty suffix_list_urls). When
    `expected_sha256` is given, a mismatch raises instead of returning:
    the PSL check is a build gate, not a report someone must remember to
    read.
    """
    sha = gate_psl_snapshot(expected_sha256)
    rows = []
    for u in urls:
        h = host_of(u)
        reg = registrable_domain(h)
        key, hosted = cache_key(u)
        rows.append(
            {
                "url": u,
                "host": h,
                "registrable_domain": reg,
                "key": key,
                "hosted": hosted,
            }
        )
    return {
        "psl_snapshot_sha256": sha,
        "rows": rows,
    }
