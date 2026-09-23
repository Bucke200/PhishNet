# Chrome Extension Packaging & Production Deployment Guide

This guide documents the end-to-end engineering process for transforming the PhishNet browser extension from an **unpacked local development directory** into a **production-packed Chrome Web Store release**.

---

## 1. Extension Distribution Modes

Before packaging, it is essential to understand the three distribution formats in Google Chrome:

| Mode | Format | Distribution Channel | ID Stability | Use Case |
|---|---|---|---|---|
| **Unpacked** | Folder (`extension/`) | `chrome://extensions` > "Load unpacked" | Pinned via `"key"` in `manifest.json` | Local development, debugging, rapid reloading. |
| **Locally Packed** | `.crx` + `.pem` key | Manual file download | Pinned via private `.pem` key | **Blocked by Chrome for standard users** unless Developer Mode or Enterprise GPO is enabled. Not viable for public distribution. |
| **Chrome Web Store** | `.zip` archive | [Chrome Web Store](https://chromewebstore.google.com) | Assigned permanently by Google on upload | **Official public distribution.** Auto-updates silently, trusted by Chrome, zero security warnings. |

---

## 2. Pre-Packaging Requirements & Manifest Compliance

Google enforces strict validation rules when an archive is uploaded to the Chrome Web Store:

### 2.1 Strip the `"key"` Field
- **Problem**: In local development, `manifest.json` contains `"key": "MIIBIjAN..."` so Chrome assigns the predictable ID `cphacgebncakdmjbpoibajnihhbbcjec`.
- **CWS Policy**: Google **rejects** ZIP uploads that contain `"key"` for new items:
  `"Could not load manifest. Field 'key' is not allowed in Web Store packages."`
- **Solution**: The root `extension/manifest.json` retains `"key"` for developer convenience, but the packaging build script automatically strips `"key"` when generating the distribution archive.

### 2.2 Remove Unused Permissions
- **Problem**: `extension/manifest.json` requested `"alarms"`, which is unreferenced in code.
- **CWS Policy**: Google enforces the "Least Privilege / Minimal Permissions" policy. Requesting unreferenced permissions triggers review rejection or prolonged manual audits.
- **Solution**: Removed from `permissions` (enforced by the packaging script, which fails the build if banned permissions reappear).

### 2.3 Point to Production Serving Backend
- In local development, `extension/constants.js` defaults to `http://localhost:8000`.
- In the production package, `PHISHNET_DEFAULT_BACKEND` must point to the verified Google Cloud Run serving URL:
  `https://phishnet-serving-683912591639.us-central1.run.app`

---

## 3. Automated Packaging Tool (`scripts/package_extension.py`)

A dedicated packaging script automates validation, asset preparation, and archive creation:

```powershell
# Build the production release ZIP package (Pillow is packaging-only,
# fetched ephemerally so the locked runtime stays minimal)
uv run --with pillow python scripts/package_extension.py
```

### What the packaging tool does:
1. **Manifest Validation**:
   - Confirms `manifest_version: 3`.
   - Validates description length (< 132 characters).
   - Verifies that unused permissions (e.g. `alarms`) are absent.
2. **Manifest Sanitization**:
   - Strips `"key"` from `manifest.json` for the distribution copy.
   - Bakes the production backend URL into `constants.js`.
3. **Asset Generation**:
   - Generates exact 128x128 store icon (`dist/store_assets/store_icon_128.png`).
   - Generates 440x280 small promotional tile (`dist/store_assets/promo_tile_440x280.png`).
   - Formats screenshots to 1280x800 px in `dist/store_assets/`.
4. **Archive Packaging**:
    - Emits a clean, lightweight `.zip` archive named from the manifest
      version (`dist/phishnet-extension-v2.0.zip` for manifest `2.0`).
    - Excludes `.git`, `.DS_Store`, editor caches, and development-only files.
5. **Checksum & Pre-flight Report**:
   - Calculates and prints the SHA-256 hash of the release package for verification.

---

## 4. Backend Cloud Run Alignment (CORS)

Before publishing, ensure the Cloud Run serving backend is prepared to accept requests from the new Web Store Extension ID.

### 4.1 Hybrid CORS Architecture
In `src/phishnet/serving/app.py`, CORS allows localhost plus Chromium
extension origins (pinned IDs when `PHISHNET_EXTENSION_ID` lists them,
otherwise any 32-char extension ID via regex — Starlette `fullmatch`):
```python
origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
ext_raw = extension_id or os.getenv("PHISHNET_EXTENSION_ID", "")
allow_origin_regex = None
if ext_raw and ext_raw.strip().lower() not in ("*", "any", "all"):
    for ext in [e.strip() for e in ext_raw.split(",") if e.strip()]:
        origins.append(f"chrome-extension://{ext}")
else:
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

### 4.2 Updating Cloud Run Deployment
Deploy with `PHISHNET_EXTENSION_ID=*` (regex mode — the new store ID works
with no further backend changes):
```bash
PHISHNET_EXTENSION_ID=* ./deploy/deploy_cloudrun.sh
```
Or update the live service directly:
```bash
gcloud run services update phishnet-serving \
  --region=us-central1 \
  --update-env-vars="PHISHNET_EXTENSION_ID=*"
```

---

## 5. Chrome Web Store Submission Walkthrough

### Step 1: Open Developer Dashboard
Navigate to [https://chrome.google.com/webstore/devconsole](https://chrome.google.com/webstore/devconsole).
- Sign in with your Google Account (ensure 2-Step Verification is active).
- Pay the one-time $5 developer registration fee if prompt appears.

### Step 2: Upload Package
1. Click **+ New Item**.
2. Drag and drop `dist/phishnet-extension-v2.0.zip`.
3. Once processed, note the assigned **Item ID** in the console URL.

### Step 3: Fill Store Listing Details
Copy the pre-approved text from [`docs/webstore-listing.md`](webstore-listing.md):
- **Item Title**: `PhishNet Detector`
- **Summary**: `Real-time machine learning phishing detection with explainable AI verdicts directly in your browser.`
- **Detailed Description**: Paste the Markdown overview from `docs/webstore-listing.md`.
- **Category**: `Productivity` > `Tools`.
- **Language**: `English`.

### Step 4: Upload Graphic Assets
Upload the generated files from `dist/store_assets/`:
- **Store Icon**: `store_icon_128.png` (128x128 px)
- **Small Promo Tile**: `promo_tile_440x280.png` (440x280 px)
- **Screenshots**: Upload at least 1 screenshot (1280x800 px).

### Step 5: Complete Privacy Practices Tab
Google strictly evaluates extensions accessing web URLs:
1. **Single Purpose**: Paste the single-purpose description from `docs/webstore-listing.md`.
2. **Permission Justifications**:
   - `tabs`: Required to observe URL navigations for real-time risk classification.
   - `notifications`: Required to display warning/safe alerts.
   - `storage`: Required to store user preferences locally.
   - `host_permissions`: Required to communicate with Cloud Run serving API over HTTPS.
3. **Data Usage**:
   - Check **Web History** (URLs visited, ephemeral scoring only).
   - Check all 4 certification boxes (Limited Use compliance, no data sale, single-purpose only, no creditworthiness).
4. **Privacy Policy URL**:
   Enter: `https://bucke200.github.io/PhishNet/privacy.html`

### Step 6: Submit for Review
Click **Submit for review**. Review typically takes 24–72 hours for extensions with `"tabs"` permissions.

---

## 6. Synchronizing Local Development with Web Store ID

Once your extension is uploaded to the Chrome Developer Dashboard, you can make your local unpacked extension share the **exact same ID** as the Web Store release:

1. In the Chrome Developer Dashboard, click on your extension item.
2. Navigate to the **Package** tab on the left sidebar.
3. Click **"View public key"**.
4. Copy the long public key string between `-----BEGIN PUBLIC KEY-----` and `-----END PUBLIC KEY-----` (ensure all newlines are removed so it forms a single line).
5. Open `extension/manifest.json` in your editor and replace the `"key"` value with this copied key:
   ```json
   "key": "<COPIED_PUBLIC_KEY_FROM_WEBSTORE_DASHBOARD>"
   ```
6. Reload the extension in `chrome://extensions`.
7. Your local unpacked extension and the official Web Store extension now share the exact same 32-letter extension ID forever.
