# PhishNet Detector — Privacy Policy

*Last Updated: September 23, 2026*  
*Chrome Web Store Compliance*

---

## Core Privacy Guarantee
**PhishNet Detector operates on a strict minimal-data principle.** We do not track users, do not record browsing histories, do not store personal data, and never sell or monetize user information.

---

## 1. Single Purpose & Scope
**PhishNet Detector** is an open-source cybersecurity browser extension designed with a single, dedicated purpose: to protect users from malicious phishing attacks and credential harvesting by classifying visited URLs in real time using machine learning models.

---

## 2. Information We Collect and Process
To detect phishing sites, the extension processes the following minimal information:

- **Page URL (Web Address):** When a user navigates to an active tab in Google Chrome, the URL (e.g. `https://example.com/login`) is captured by the extension's service worker to perform risk scoring.
- **User Preferences:** Custom configuration options entered in the extension's Options page (such as a custom self-hosted backend URL) are stored exclusively in your browser's local storage (`chrome.storage.local`). This data remains on your device and is never sent to us.

---

## 3. How We Use the Information
The visited URL is transmitted securely over an encrypted HTTPS connection to the PhishNet scoring backend API:

1. The URL is parsed into structural lexical features (e.g., domain characteristics, path depth, character entropy).
2. The model calculates a phishing probability score and SHAP feature attributions.
3. The disposition verdict (`alert`, `allow`, or `can't assess`) is returned to the extension to display a real-time notification to the user.

---

## 4. Ephemeral Processing & Zero Data Retention
- **No Identity Association:** URLs sent for scoring are completely decoupled from your identity. The extension does not collect names, email addresses, device fingerprints, or account credentials.
- **No Persistent Browsing History:** The PhishNet backend performs ephemeral in-memory scoring. It does not maintain a database of URLs visited by individual users. Operational logs record only the scored hostname, score band, and outcome for reliability monitoring (Google Cloud Logging, default 30-day retention); platform-level request logs are retained per Google Cloud defaults.
- **No Cookies or User Profiling:** PhishNet does not set advertising cookies, tracking pixels, or user behavior analytics.

---

## 5. Third-Party Sharing and Commercialization
We believe your browsing privacy is non-negotiable:
- We **do not sell, rent, or trade** user data to any third party, broker, or advertiser.
- We **do not use** user data for advertising, marketing, retargeting, or creditworthiness evaluations.
- We **do not transfer** data to external third parties, except:
  - **Groq inference API (`api.groq.com`):** when an in-band page needs a judgment, its canonical page extract (title, visible text, form structure, link hosts — never passwords or entered values) is sent for scoring under Groq's API data policy.
  - **Sandboxed fetcher infrastructure:** dynamic headless page verification when in-band risk thresholds are triggered.

---

## 6. Permissions Justification
PhishNet requests only the minimum permissions necessary to function:
- `tabs`: Required to detect when a tab completes navigation so the URL can be checked for phishing before user interaction.
- `notifications`: Required to display real-time security alerts and safety verdicts directly on the user's desktop.
- `storage`: Required to persist local preferences (e.g. backend endpoint) on the user's device.
- `host_permissions`: Required to communicate exclusively with the PhishNet scoring backend over secure HTTPS.

---

## 7. User Rights and Controls
You maintain complete control over the extension:
- You may inspect the full open-source codebase at any time.
- You can configure custom self-hosted backend endpoints in the extension Options page.
- You can temporarily disable or uninstall the extension at any time via `chrome://extensions`, immediately terminating all URL processing.

---

## 8. Contact & Open Source Verification
PhishNet is an open-source project. If you have any questions or feedback regarding this Privacy Policy, please open an issue or reach out:
- **Source Code Repository:** [https://github.com/Bucke200/PhishNet](https://github.com/Bucke200/PhishNet)
- **Maintainer Contact:** [srinjaypanja200@gmail.com](mailto:srinjaypanja200@gmail.com)
