# Chrome Web Store Developer Console Submission Guide & Assets

This guide contains the exact copy-paste text, permission justifications, privacy disclosures, and asset specifications required when submitting **PhishNet Detector** to the [Chrome Web Store Developer Console](https://chrome.google.com/webstore/devconsole).

---

## 1. Product Details Tab

### Item Title
```text
PhishNet Detector
```
*(17 / 45 characters)*

### Summary / Short Description
```text
Real-time machine learning phishing detection with explainable AI verdicts directly in your browser.
```
*(100 / 132 characters)*

### Detailed Description
```markdown
PhishNet Detector protects you from credential harvesting, fraudulent websites, and advanced phishing attacks using a high-speed, two-stage machine learning cascade.

Unlike traditional blocklist-only security tools that fail against zero-day phishing domains, PhishNet scores web addresses in real time before you interact with them, providing immediate visual alerts and explainable feature attributions.

KEY FEATURES
⚡ Sub-Millisecond Inference: High-speed Tier-1 LightGBM inference delivers sub-millisecond lexical scoring without slowing down your browsing experience.
🔍 Explainable AI (Native SHAP): Understand WHY a site was flagged. PhishNet extracts and displays the top mathematical feature attributions (domain entropy, path depth, suspicious tokens, etc.) directly in security notifications.
🛡️ Dynamic Cascade Sandbox: For ambiguous URLs, an automated serverless headless browser analyzes page DOM characteristics and WAF behaviors to defeat anti-bot and cloaking defenses.
🔒 Privacy First & Zero Data Retention: Your browsing history is never recorded, sold, or profiled. URL scoring occurs ephemerally in-memory over encrypted HTTPS.
⚙️ Fully Configurable: Open-source architecture allows developers and enterprise operators to connect the extension to their own self-hosted PhishNet serving containers.

HOW IT WORKS
1. When you navigate to a webpage, PhishNet's lightweight service worker intercepts the URL.
2. The URL is evaluated against the PhishNet scoring model.
3. If the URL is safe, you browse uninterrupted (or see a safe-confirmation notification).
4. If the URL is flagged as phishing, an immediate warning notification alerts you before credentials can be entered.

TRANSPARENCY & OPEN SOURCE
PhishNet is an open-source security project. Review our complete architecture, model cards, and source code:
https://github.com/Bucke200/PhishNet
```

### Category
```text
Productivity > Tools
```
*(Alternative: Utilities / Security)*

### Language
```text
English
```

---

## 2. Privacy Practices Tab

Google's review team places high scrutiny on extensions that inspect web navigation (`tabs` permission). Use the exact justifications below:

### Single Purpose Description
```text
PhishNet Detector protects users from phishing attacks by performing real-time URL risk classification and displaying instant security warnings with explainable machine learning feature attributions.
```

### Permission Justifications

#### `tabs`
```text
Required to detect navigation events and extract the visited webpage URL so it can be evaluated against the PhishNet phishing detection model before the user enters sensitive credentials.
```

#### `notifications`
```text
Required to display immediate desktop alert notifications to the user whenever a visited URL is identified as phishing, suspicious, or verified safe.
```

#### `storage`
```text
Required to save user preferences locally in the browser, including custom self-hosted backend API endpoint URLs and display configurations.
```

#### `host_permissions` (`https://phishnet-serving-683912591639.us-central1.run.app/*`)
```text
Required to communicate securely over HTTPS with the PhishNet model serving API to request phishing probability scores and tree-SHAP explanations.
```

### Data Usage Certifications

When prompted with Google's data disclosure questionnaire:

1. **Does your extension collect user data?**
   - Select: **Yes**
2. **Data Types:**
   - Under **Personal Information**: Check **Web History** (Specifically: *"URLs visited, exclusively for real-time security and phishing risk classification"*).
3. **Certifications (Check all 4 boxes):**
   - [x] **I certify that my extension complies with the Limited Use policy.**
   - [x] **I certify that user data is not sold to third parties.**
   - [x] **I certify that user data is not used or transferred for purposes unrelated to the extension's single purpose.**
   - [x] **I certify that user data is not used or transferred to determine creditworthiness or for lending purposes.**

### Privacy Policy URL
```text
https://bucke200.github.io/PhishNet/privacy.html
```

---

## 3. Store Visual Assets Checklist

Prepare the following images in the **Store Listing > Graphic assets** section:

| Asset | Exact Dimensions | Format | Requirement | Location / Generator |
|---|---|---|---|---|
| **Store Icon** | 128 x 128 px | PNG (transparent) | **Mandatory** | `dist/store_assets/store_icon_128.png` |
| **Small Promo Tile** | 440 x 280 px | PNG or JPEG | **Mandatory** | `dist/store_assets/promo_tile_440x280.png` |
| **Screenshots** | 1280 x 800 px (or 640 x 400 px) | PNG or JPEG | **At least 1 mandatory** (up to 5) | `dist/store_assets/screenshot1.png` |
| **Marquee Promo Tile** | 1400 x 560 px | PNG or JPEG | Optional (featured placement) | — |

---

## 4. Enabling the Privacy Policy on GitHub Pages (2 Minutes)

To activate `https://bucke200.github.io/PhishNet/privacy.html`:

1. Commit and push `privacy.html` (kept at the repo root — `docs/` holds Markdown only) to your GitHub repository:
   ```bash
   git add privacy.html docs/webstore-listing.md
   git commit -m "docs: add Chrome Web Store privacy policy and listing guide"
   git push origin master
   ```
2. In your web browser, navigate to your repository settings:
   `https://github.com/Bucke200/PhishNet/settings/pages`
3. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`
   - **Branch**: Select `master`
   - **Folder**: Select `/ (root)`
   - Click **Save**.
4. GitHub Pages will build the site in approximately 60 seconds. Verify by opening:
   `https://bucke200.github.io/PhishNet/privacy.html`

---

## 5. Chrome Web Store Upload & Release Steps

1. **Package the Extension**:
   ```powershell
   uv run --with pillow python scripts/package_extension.py
   ```
   This generates `dist/phishnet-extension-v2.0.zip` with the `"key"` stripped and assets verified.

2. **Open Developer Dashboard**:
   Visit [https://chrome.google.com/webstore/devconsole](https://chrome.google.com/webstore/devconsole).
   *(If this is your first time, complete the one-time $5 developer registration fee)*.

3. **Upload ZIP**:
   Click **+ New Item** and drag-and-drop `dist/phishnet-extension-v2.0.zip`.

4. **Copy the New Item ID**:
   As soon as the upload completes, note your new 32-character extension ID shown in the dashboard URL.

5. **Fill in the Tabs**:
   - Copy the text from **§1 Product Details** above.
   - Copy the justifications from **§2 Privacy Practices** above.
   - Upload the graphics from `dist/store_assets/` (Icon, Tile, Screenshots).

6. **Submit for Review**:
   Click **Submit for review**. Review typically takes 24–72 hours for extensions with `"tabs"` permissions.
