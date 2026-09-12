"""Append-only daily collection of phishing and benign URLs.

    python collect.py --phish --benign --benign-domains 300

Writes one JSONL file per source per day under data/raw/. Never rewrites
history — the append-only log is what makes the temporal split real.

Why a daily cron and not a one-shot download:

  The OpenPhish community feed is a snapshot of what is live right now, with no
  timestamps. You cannot temporally split a feed that has no time axis. Running
  this daily gives every URL a first_observed date that is at worst an upper
  bound on its true first-seen, and consistently so for every row. Start the cron
  on day 1 of the week; by the time the model work begins you have a usable
  window. PhishTank's online-valid dump does carry submission_time, so those rows
  get a real date immediately — use it to backfill.

Benign side: Tranco gives a permanent, citable list ID (not "the top 1M as of
whenever"), and we crawl real internal links rather than shipping bare domains,
because bare-domain-vs-deep-URL is the exact artifact that made the old model
look good.

Two benign modes:

  --tranco-id ABC123   use exactly that historical/permanent list (reproducible)
  --tranco-latest      resolve the current daily list via the Tranco API,
                       pin its permanent list ID, then download that exact list

--tranco-latest reads its credential from the TRANCO_API_KEY environment
variable (never from the command line, so it cannot leak via process lists
or shell history). Per the Tranco API docs the API uses HTTP Basic Auth
with the account email as username and the API token as password; the email
is optionally read from TRANCO_ACCOUNT_EMAIL. Every benign row records the
exact list ID used, so "latest at collection time" stays reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import sys
import time
import urllib.robotparser as robotparser
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

RAW = Path("data/raw")
UA = "PhishNetResearchBot/0.1 (+https://github.com/Bucke200/PhishNet; academic project)"
OPENPHISH = "https://openphish.com/feed.txt"
PHISHTANK = "http://data.phishtank.com/data/online-valid.json.gz"
# Pin the Tranco list ID for reproducibility. Get one from https://tranco-list.eu
# and record it in the report; "top 1M" without an ID is not reproducible.
TRANCO = "https://tranco-list.eu/download/{list_id}/1000000"
# Tranco API (https://tranco-list.eu/api_documentation): resolving the current
# daily list returns metadata including the permanent list_id plus a download
# URL for that exact list. Only the ID is ever stored; credentials never are.
TRANCO_API_BASE = "https://tranco-list.eu/api"
TRANCO_API_LATEST = f"{TRANCO_API_BASE}/lists/date/latest"
TRANCO_API_KEY_ENV = "TRANCO_API_KEY"
TRANCO_API_EMAIL_ENV = "TRANCO_ACCOUNT_EMAIL"


def _write(rows: list[dict], source: str, today: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{source}-{today}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"{source}: {len(rows):,} rows -> {path}", file=sys.stderr)
    return path


def fetch_openphish(today: str) -> list[dict]:
    r = requests.get(OPENPHISH, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        {
            "url": u.strip(),
            "label": 1,
            "first_seen": stamp,
            "source": "openphish",
            "time_basis": "observed",
        }
        for u in r.text.splitlines()
        if u.strip().startswith("http")
    ]


def fetch_phishtank(today: str, app_key: str | None) -> list[dict]:
    url = (
        PHISHTANK
        if not app_key
        else f"http://data.phishtank.com/data/{app_key}/online-valid.json.gz"
    )
    r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    data = json.loads(gzip.decompress(r.content))
    out = []
    for e in data:
        if not e.get("url"):
            continue
        out.append(
            {
                "url": e["url"],
                "label": 1,
                # Real submission time — this is the only genuinely dated source.
                "first_seen": e.get("submission_time")
                or datetime.now(timezone.utc).isoformat(),
                "source": "phishtank",
                "time_basis": "submitted",
                "target": e.get("target"),
                "verified": e.get("verified"),
            }
        )
    return out


class TrancoError(RuntimeError):
    """The Tranco list could not be resolved or downloaded."""


@dataclass
class TrancoListRef:
    """A pinned Tranco list: permanent ID plus how to fetch that exact list."""

    list_id: str
    download_url: str
    created_on: str | None
    resolved_at: str


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Tranco downloads have been observed in two shapes: a ZIP archive holding
# the rank/domain CSV, and the plain rank/domain CSV itself. Detect the real
# payload via magic bytes rather than trusting headers.
ZIP_MAGIC = b"PK\x03\x04"


def _parse_tranco_archive(content: bytes, n: int) -> list[str]:
    """Parse a downloaded Tranco list (ZIP or plain CSV) into top-n domains."""
    if not content:
        raise TrancoError("Tranco list download was empty")
    if content[:4] == ZIP_MAGIC:
        return _parse_tranco_zip(content, n)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise TrancoError(f"could not decode Tranco list as text: {e}") from e
    return _parse_tranco_csv(text, n)


def _parse_tranco_zip(content: bytes, n: int) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = z.namelist()[0]
            lines = z.read(name).decode().splitlines()
    except Exception as e:
        raise TrancoError(f"could not parse Tranco list archive: {e}") from e
    domains = [ln.split(",", 1)[1].strip() for ln in lines[:n] if "," in ln]
    if not domains:
        raise TrancoError("Tranco list archive contained no domains")
    return domains


def _parse_tranco_csv(text: str, n: int) -> list[str]:
    """Parse plain `rank,domain` CSV text; blank/malformed rows are skipped."""
    ranked: list[tuple[int, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rank_raw, sep, domain = line.partition(",")
        if not sep:
            continue
        domain = domain.strip()
        try:
            rank = int(rank_raw.strip())
        except ValueError:
            continue
        if not domain:
            continue
        ranked.append((rank, domain))
    if not ranked:
        raise TrancoError("Tranco list CSV contained no valid rank,domain rows")
    return [domain for _, domain in ranked[:n]]


def resolve_tranco_latest(
    api_key: str, email: str | None = None, timeout: int = 60
) -> TrancoListRef:
    """Resolve the current Tranco daily list to its permanent list ID.

    Makes exactly one "latest" metadata request and pins the returned ID.
    Callers must download that exact list (see download_tranco_list) rather
    than asking for "latest" again, so the collection stays reproducible.
    Never logs or stores the API key.
    """
    # Per the Tranco API docs: HTTP Basic Auth, account email as username
    # and API token as password. The metadata endpoint answers with
    # {list_id, available, download, created_on, ...}.
    auth = HTTPBasicAuth(email or "", api_key)
    try:
        r = requests.get(
            TRANCO_API_LATEST, auth=auth, headers={"User-Agent": UA}, timeout=timeout
        )
    except requests.RequestException as e:
        raise TrancoError(f"Tranco API request failed: {e}") from e
    if r.status_code != 200:
        raise TrancoError(
            f"Tranco API returned HTTP {r.status_code} for {TRANCO_API_LATEST}"
        )
    try:
        meta = r.json()
    except ValueError as e:
        raise TrancoError(f"Tranco API returned a non-JSON response: {e}") from e
    if not isinstance(meta, dict):
        raise TrancoError("Tranco API returned an unexpected response (not an object)")
    list_id = meta.get("list_id")
    if not isinstance(list_id, str) or not list_id:
        raise TrancoError("Tranco API response did not contain a permanent list_id")
    if not meta.get("available", True):
        raise TrancoError(f"Tranco list {list_id} is not available yet")
    download = meta.get("download")
    if not isinstance(download, str) or not download:
        download = TRANCO.format(list_id=list_id)
    created = meta.get("created_on")
    return TrancoListRef(
        list_id=list_id,
        download_url=download,
        created_on=created if isinstance(created, str) else None,
        resolved_at=_utcnow(),
    )


def download_tranco_list(ref: TrancoListRef, n: int, timeout: int = 300) -> list[str]:
    """Download the exact pinned list (one request, no re-resolution)."""
    try:
        r = requests.get(ref.download_url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise TrancoError(f"Tranco list download failed: {e}") from e
    return _parse_tranco_archive(r.content, n)


def fetch_tranco(list_id: str, n: int) -> list[str]:
    r = requests.get(
        TRANCO.format(list_id=list_id), headers={"User-Agent": UA}, timeout=300
    )
    r.raise_for_status()
    return _parse_tranco_archive(r.content, n)


def _allowed(domain: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(f"https://{domain}/robots.txt")
    try:
        rp.read()
    except Exception:
        return True  # no robots.txt reachable -> treat as unrestricted
    return rp.can_fetch(UA, f"https://{domain}/")


def crawl_domain(domain: str, per_domain: int, timeout: int = 12) -> list[str]:
    """Collect real internal deep links from one domain's homepage."""
    if not _allowed(domain):
        return []
    try:
        r = requests.get(
            f"https://{domain}/",
            headers={"User-Agent": UA},
            timeout=timeout,
            allow_redirects=True,
        )
        if r.status_code >= 400 or "text/html" not in r.headers.get("content-type", ""):
            return []
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []

    base = r.url
    host = urlparse(base).netloc
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not isinstance(href, str):
            continue
        u = urljoin(base, href)
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc != host:
            continue
        if not p.path.strip("/"):
            continue  # this is the bare domain again
        found.add(p._replace(fragment="").geturl())
    picked = sorted(found)
    random.Random(domain).shuffle(picked)
    return picked[:per_domain]


def collect_benign(
    list_id: str, n_domains: int, per_domain: int, workers: int, today: str
) -> list[dict]:
    domains = fetch_tranco(list_id, n_domains)
    return collect_benign_from_domains(domains, list_id, per_domain, workers, today)


def collect_benign_from_domains(
    domains: list[str],
    list_id: str,
    per_domain: int,
    workers: int,
    today: str,
    tranco_resolved_at: str | None = None,
) -> list[dict]:
    """Crawl an already-fetched domain list; every row pins the exact list ID."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(crawl_domain, d, per_domain): d for d in domains}
        for i, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            links = fut.result()
            # Keep the homepage too, but only one per domain, so the negative
            # class is not dominated by bare domains.
            rows.append(
                {
                    "url": f"https://{d}/",
                    "label": 0,
                    "first_seen": stamp,
                    "source": f"tranco:{list_id}",
                    "time_basis": "crawled",
                    "tranco_list_id": list_id,
                    "tranco_resolved_at": tranco_resolved_at,
                    "tranco_rank": domains.index(d) + 1,
                }
            )
            for u in links:
                rows.append(
                    {
                        "url": u,
                        "label": 0,
                        "first_seen": stamp,
                        "source": f"tranco:{list_id}",
                        "time_basis": "crawled",
                        "tranco_list_id": list_id,
                        "tranco_resolved_at": tranco_resolved_at,
                        "tranco_rank": domains.index(d) + 1,
                    }
                )
            if i % 25 == 0:
                print(
                    f"  crawled {i}/{len(domains)} domains, {len(rows):,} urls",
                    file=sys.stderr,
                )
            time.sleep(0.05)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phish", action="store_true")
    p.add_argument("--benign", action="store_true")
    p.add_argument("--phishtank-key", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--tranco-id", default="NONE", help="pinned Tranco list ID, e.g. K2XVW"
    )
    g.add_argument(
        "--tranco-latest",
        action="store_true",
        help="resolve the current list via the Tranco API "
        f"(requires {TRANCO_API_KEY_ENV})",
    )
    p.add_argument("--benign-domains", type=int, default=300)
    p.add_argument("--per-domain", type=int, default=8)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    today = date.today().isoformat()
    if a.phish:
        _write(fetch_openphish(today), "openphish", today)
        try:
            _write(fetch_phishtank(today, a.phishtank_key), "phishtank", today)
        except Exception as e:
            print(
                f"phishtank fetch failed ({e}); openphish snapshot still written",
                file=sys.stderr,
            )
    if a.benign:
        if a.tranco_latest:
            return _collect_benign_latest(a, today)
        if a.tranco_id == "NONE":
            print(
                "refusing to crawl without a Tranco list version; "
                "use --tranco-id with a pinned list ID, or --tranco-latest "
                f"with {TRANCO_API_KEY_ENV} set",
                file=sys.stderr,
            )
            return 2
        _write(
            collect_benign(
                a.tranco_id, a.benign_domains, a.per_domain, a.workers, today
            ),
            "benign",
            today,
        )
    return 0


def _collect_benign_latest(a: argparse.Namespace, today: str) -> int:
    """Automated mode: resolve the current list once, then crawl that exact list."""
    api_key = os.environ.get(TRANCO_API_KEY_ENV)
    if not api_key:
        print(
            "refusing to resolve the latest Tranco list without an API key; "
            f"set the {TRANCO_API_KEY_ENV} environment variable "
            "(or use --tranco-id with a pinned list ID)",
            file=sys.stderr,
        )
        return 2
    try:
        ref = resolve_tranco_latest(api_key, email=os.environ.get(TRANCO_API_EMAIL_ENV))
    except TrancoError as e:
        print(
            f"tranco latest-list resolution failed ({e}); "
            "refusing to collect from an unknown list",
            file=sys.stderr,
        )
        return 1
    print(
        f"resolved latest Tranco list {ref.list_id} "
        f"(created {ref.created_on}; resolved {ref.resolved_at})",
        file=sys.stderr,
    )
    try:
        domains = download_tranco_list(ref, a.benign_domains)
    except TrancoError as e:
        print(f"tranco list download failed ({e})", file=sys.stderr)
        return 1
    _write(
        collect_benign_from_domains(
            domains,
            ref.list_id,
            a.per_domain,
            a.workers,
            today,
            tranco_resolved_at=ref.resolved_at,
        ),
        "benign",
        today,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
