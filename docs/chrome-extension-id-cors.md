# Chrome Extension ID & Cloud Run CORS Architecture Review

## 1. Executive Summary

When transitioning the PhishNet browser extension from an unpacked local development installation into a published Chrome Web Store extension, a critical production failure mode was identified:

**The serving backend's CORS configuration hardcodes a single local extension ID (`cphacgebncakdmjbpoibajnihhbbcjec`). Because Google assigns a new, unpredictable extension ID upon store publication, every user who installs the extension from the Chrome Web Store will experience immediate CORS failures, completely breaking scoring and notifications.**

This document details the root cause, live production verification, why the hardcoding existed, and the permanent hybrid CORS solution.

---

## 2. Root Cause Analysis

### 2.1 The Extension ID Derivation
In Manifest V3, an extension's ID is the 32-character lowercase hex representation (letters `a` through `p`) of the SHA-256 hash of its public RSA key.

- In `extension/manifest.json`, the `"key"` property contains a development RSA public key.
- Chrome uses this key to deterministically compute:
  ```text
  cphacgebncakdmjbpoibajnihhbbcjec
  ```
- Any browser loading the unpacked directory receives this exact ID.

### 2.2 The Chrome Web Store Conflict
- Google Developer Program Policies **strictly forbid** developer-provided `"key"` fields in uploaded ZIP archives for new store items (`"Field 'key' is not allowed in Web Store packages"`).
- When a developer uploads an extension to the Chrome Web Store Developer Dashboard, Google generates an internal cryptographic keypair and assigns a **new, permanent 32-letter Extension ID** (e.g., `pbaodkfe...`).

### 2.3 The Cloud Run CORS Failure
The FastAPI backend ([`src/phishnet/serving/app.py`](../src/phishnet/serving/app.py)) currently initializes CORS as follows:

```python
origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
ext = extension_id or os.getenv("PHISHNET_EXTENSION_ID")
if ext:
    origins.append(f"chrome-extension://{ext}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

And Cloud Run was deployed with:
```bash
PHISHNET_EXTENSION_ID=cphacgebncakdmjbpoibajnihhbbcjec
```

When a published user navigates to a webpage:
1. The extension background worker sends `fetch("https://phishnet-serving-...run.app/predict")`.
2. Chrome automatically attaches the request header:
   `Origin: chrome-extension://<NEW_CWS_ID>`
3. FastAPI's `CORSMiddleware` checks `allow_origins`. The origin does not match `chrome-extension://cphacgebncakdmjbpoibajnihhbbcjec`.
4. The response omits the `Access-Control-Allow-Origin` header.
5. The user's browser sandbox intercepts the response and throws:
   ```text
   Access to fetch at 'https://.../predict' from origin 'chrome-extension://<NEW_ID>' 
   has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present.
   ```
6. **Result**: The extension silently drops all predictions, fails to notify, and shows errors in the console.

---

## 3. Live Production Verification

We executed a live verification test against the active Cloud Run serving endpoint (`https://phishnet-serving-683912591639.us-central1.run.app/predict`):

```python
# Test script verifying Access-Control-Allow-Origin response header
Origin: chrome-extension://cphacgebncakdmjbpoibajnihhbbcjec
  Status: 200
  Access-Control-Allow-Origin: chrome-extension://cphacgebncakdmjbpoibajnihhbbcjec  <-- (Allowed)

Origin: chrome-extension://differentrandomidhereabcdefghijkl
  Status: 200
  Access-Control-Allow-Origin: None                                                 <-- (BLOCKED)

Origin: https://evil-website.com
  Status: 200
  Access-Control-Allow-Origin: None                                                 <-- (BLOCKED)
```

The test proved that any extension ID other than the local dev ID is rejected by the server's CORS policy.

---

## 4. Why Was It Hardcoded?

During Phase 6 development, the hardcoded ID was introduced to prevent unauthorized third-party websites (`https://...`) from hotlinking the PhishNet inference API.

However:
1. CORS is an in-browser security mechanism. It does not stop command-line tools (`curl`, Python) from invoking public endpoints without an `Origin` header.
2. The PhishNet API is stateless and carries no cookies or credentials (`allow_credentials=False`).
3. The true security objective is:
   - **BLOCK** cross-origin calls originating from web pages (`https://*`, `http://*`).
   - **ALLOW** legitimate Chromium extensions (`chrome-extension://*`).

Pinning a single 32-character hash created an unnecessary chicken-and-egg deployment bottleneck.

---

## 5. The Solution: Resilient Hybrid CORS Architecture

Instead of hardcoding a single static ID, the backend is upgraded to support **hybrid origin evaluation** in `src/phishnet/serving/app.py`:

```python
    # CORS: Allow localhost and Chrome extensions.
    # Regular websites (https://evil.com) remain strictly BLOCKED.
    origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    ext_raw = extension_id or os.getenv("PHISHNET_EXTENSION_ID", "")

    allow_origin_regex = None
    if ext_raw and ext_raw.strip().lower() not in ("*", "any", "all"):
        # Explicit allowlist of specific extension IDs (comma-separated)
        for ext in [e.strip() for e in ext_raw.split(",") if e.strip()]:
            origins.append(f"chrome-extension://{ext}")
    else:
        # Resilient: matches ANY Chromium extension ID (Starlette fullmatch,
        # so the unanchored pattern is exact).
        allow_origin_regex = r"chrome-extension://[a-z]{32}"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
```

### Architectural Guarantees:
1. **Zero-Breakage**: Any valid Chromium extension (local development, Web Store release, beta testers, Edge Add-ons) receives a valid `Access-Control-Allow-Origin` header automatically.
2. **Web Isolation**: Malicious websites (`https://...`) are still denied CORS authorization.
3. **Enterprise Pinning**: Operators who require strict pinning can supply comma-separated IDs via `PHISHNET_EXTENSION_ID="id1,id2"`.

---

## 6. Immediate Cloud Run Remediation Runbook

If you want to update the currently deployed Cloud Run service without rebuilding images, you can update the environment variable to allow both the dev ID and the new Web Store ID once assigned:

```bash
# 1. Update Cloud Run service with comma-separated IDs
gcloud run services update phishnet-serving \
  --region=us-central1 \
  --update-env-vars="PHISHNET_EXTENSION_ID=cphacgebncakdmjbpoibajnihhbbcjec,<NEW_STORE_ID>"

# 2. Or set to wildcard after deploying the hybrid CORS update
gcloud run services update phishnet-serving \
  --region=us-central1 \
  --update-env-vars="PHISHNET_EXTENSION_ID=*"
```
