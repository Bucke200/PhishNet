"""Model-visible page extract (extract-only, §5).

The extract is what the model sees; raw HTML is never sent. Pure function of
`(html, page_url)`: title, meta description, visible text truncated at 6,000
characters, form fields (name/type/placeholder/action host), link-host
frequency table capped at 20, iframe and script source hosts, image alt text,
declared language, favicon host. No network, no JS execution.

`canonical_extract` is the frozen serialization whose sha256 is stored beside
`sha256(raw_html)`; `to_model_text` renders the delimited untrusted block the
prompt wraps in `<untrusted_page_extract>`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

TEXT_CHAR_CAP = 6000
LINK_HOST_CAP = 20
EVIDENCE_ALT_CAP = 20


def _host_of(raw: str, base: str) -> str:
    try:
        joined = urljoin(base, raw)
        return (urlparse(joined).hostname or "").lower()
    except Exception:
        return ""


def canonical_extract(html: str, page_url: str) -> dict:
    """Deterministic extract dict (keys sorted at serialization)."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    meta_desc = ""
    for tag in soup.find_all("meta"):
        name = str(tag.get("name") or tag.get("property") or "").lower()
        if name in ("description", "og:description") and tag.get("content"):
            meta_desc = str(tag["content"]).strip()
            break

    for dead in soup(["script", "style", "noscript"]):
        dead.decompose()
    visible_text = " ".join(soup.get_text(separator=" ").split())[:TEXT_CHAR_CAP]

    forms: list[dict] = []
    for form in soup.find_all("form"):
        action = str(form.get("action") or "")
        inputs: list[dict[str, str]] = []
        for inp in form.find_all(["input", "textarea", "select", "button"]):
            inputs.append(
                {
                    "name": str(inp.get("name") or ""),
                    "type": str(inp.get("type") or inp.name or ""),
                    "placeholder": str(inp.get("placeholder") or ""),
                }
            )
        forms.append({"action_host": _host_of(action, page_url), "inputs": inputs})

    link_hosts: Counter[str] = Counter()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        if isinstance(href, str) and href.strip():
            host = _host_of(href, page_url)
            if host:
                link_hosts[host] += 1
    top_link_hosts = [
        [host, count] for host, count in link_hosts.most_common(LINK_HOST_CAP)
    ]

    iframe_hosts = sorted(
        {
            _host_of(str(t.get("src") or ""), page_url)
            for t in soup.find_all("iframe", src=True)
        }
        - {""}
    )
    script_hosts = sorted(
        {
            _host_of(str(t.get("src") or ""), page_url)
            for t in soup.find_all("script", src=True)
        }
        - {""}
    )

    alt_texts = [
        str(t.get("alt") or "").strip()
        for t in soup.find_all("img", alt=True)
        if str(t.get("alt") or "").strip()
    ][:EVIDENCE_ALT_CAP]

    lang = ""
    html_tag = soup.find("html")
    if html_tag is not None:
        lang = str(html_tag.get("lang") or "")

    favicon_host = ""
    for tag in soup.find_all("link"):
        rel_val = tag.get("rel")
        if isinstance(rel_val, str):
            rel = rel_val.lower()
        elif rel_val:
            rel = " ".join(str(v) for v in rel_val).lower()
        else:
            rel = ""
        if "icon" in rel and tag.get("href"):
            favicon_host = _host_of(str(tag["href"]), page_url)
            break

    try:
        page_host = (urlparse(page_url).hostname or "").lower()
    except Exception:
        page_host = ""

    return {
        "page_host": page_host,
        "title": title,
        "meta_description": meta_desc,
        "visible_text": visible_text,
        "forms": forms,
        "link_hosts": top_link_hosts,
        "iframe_hosts": iframe_hosts,
        "script_hosts": script_hosts,
        "image_alt_text": alt_texts,
        "language": lang,
        "favicon_host": favicon_host,
    }


def canonical_json(extract: dict) -> str:
    return json.dumps(extract, sort_keys=True, ensure_ascii=False)


def extract_hash(extract: dict) -> str:
    return hashlib.sha256(canonical_json(extract).encode("utf-8")).hexdigest()


def to_model_text(extract: dict) -> str:
    """Delimited untrusted block: the only page content the prompt carries."""
    lines = [
        f"title: {extract.get('title', '')}",
        f"meta: {extract.get('meta_description', '')}",
        f"text: {extract.get('visible_text', '')}",
    ]
    form_bits = []
    for form in extract.get("forms", []):
        fields = ",".join(
            f"{i.get('type')}:{i.get('name')}" for i in form.get("inputs", [])
        )
        form_bits.append(f"action_host={form.get('action_host', '')} [{fields}]")
    lines.append(f"forms: {'; '.join(form_bits) if form_bits else 'none'}")
    links = ",".join(f"{h}({c})" for h, c in extract.get("link_hosts", []))
    lines.append(f"link_hosts: {links if links else 'none'}")
    lines.append(f"iframe_hosts: {','.join(extract.get('iframe_hosts', [])) or 'none'}")
    lines.append(f"script_hosts: {','.join(extract.get('script_hosts', [])) or 'none'}")
    lines.append(f"language: {extract.get('language', '')}")
    lines.append(f"favicon_host: {extract.get('favicon_host', '') or 'none'}")
    body = "\n".join(lines)
    return "<untrusted_page_extract>\n" + body + "\n</untrusted_page_extract>"
