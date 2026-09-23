// PhishNet Detector - Phase 6 background service worker.
//
// Talks to the Phase 6 serving container (`phishnet.serving.app`): Tier-1
// score, fail-closed disposition, and native LightGBM SHAP. Thresholds are
// read from `/health` — never hard-coded here. The backend base URL is
// configurable from the options page (see constants.js); the
// feedback endpoint (`/report`) was removed with the feedback pipeline, so
// this worker has no write path.

try {
    importScripts("constants.js");
} catch (error) {
    console.warn("PhishNet: constants.js unavailable", error);
}
const DEFAULT_BACKEND =
    (typeof PHISHNET_DEFAULT_BACKEND !== "undefined" && PHISHNET_DEFAULT_BACKEND) ||
    "http://localhost:8000";
// Same URL within this window is not re-notified (tab re-navigation, redirects).
const NOTIFY_DEDUPE_MS = 60_000;

let HEALTH = null;
let HEALTH_BASE = null;
const recent = new Map();

async function getBackend() {
    try {
        const { backendUrl } = await chrome.storage.local.get("backendUrl");
        return (backendUrl || DEFAULT_BACKEND).replace(/\/+$/, "");
    } catch (error) {
        console.warn("PhishNet: storage unavailable, using default backend", error);
        return DEFAULT_BACKEND;
    }
}

async function getHealth(base) {
    if (HEALTH && HEALTH_BASE === base) return HEALTH;
    try {
        const response = await fetch(`${base}/health`);
        if (response.ok) {
            HEALTH = await response.json();
            HEALTH_BASE = base;
        }
    } catch (error) {
        console.warn("PhishNet: /health unavailable", error);
    }
    return HEALTH;
}

function formatScore(score) {
    return (score === null || score === undefined) ? "n/a" : Number(score).toFixed(4);
}

async function fetchExplanation(base, url) {
    try {
        const response = await fetch(`${base}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url, top_k: 3 }),
        });
        if (!response.ok) return null;
        const result = await response.json();
        return result.attribution;
    } catch (error) {
        return null;
    }
}

function describe(result, attribution) {
    const disposition = result.disposition;
    let title;
    let icon;
    if (disposition === "alert") {
        title = "Phishing warning";
        icon = "icons/icon-warning.png";
    } else if (disposition === "allow") {
        title = "Safe";
        icon = "icons/icon-safe.png";
    } else {
        // "can't assess" is not a safe verdict; use the neutral app icon.
        title = "Not assessed";
        icon = "icons/icon128.png";
    }

    const mode = result.tier2_mode || "unknown";
    const lines = [
        `${result.url}`,
        `disposition: ${disposition} (${result.reason})`,
        `Tier-1 score: ${formatScore(result.tier1_score)}`,
        `Tier 2 (${mode}): ${result.tier2 ? result.tier2.kind : "not run"}`,
    ];
    if (disposition === "allow") {
        lines.push("Below the calibrated alert band - not a safety guarantee.");
    }
    if (disposition === "can't assess") {
        lines.push("No verdict was produced - not a safety guarantee.");
    }
    if (attribution && attribution.features && attribution.features.length) {
        lines.push("Top features:");
        for (const f of attribution.features) {
            lines.push(`  ${f.feature}: ${Number(f.contribution).toFixed(3)}`);
        }
    }
    return { title, icon, message: lines.join("\n") };
}

function shouldNotify(url) {
    const now = Date.now();
    const last = recent.get(url);
    if (last && now - last < NOTIFY_DEDUPE_MS) return false;
    recent.set(url, now);
    if (recent.size > 500) {
        for (const [key, when] of recent) {
            if (now - when > NOTIFY_DEDUPE_MS) recent.delete(key);
        }
    }
    return true;
}

async function checkUrl(tabId, url) {
    if (!url || !url.startsWith("http")) return;
    if (!shouldNotify(url)) return;
    try {
        const base = await getBackend();
        await getHealth(base);
        const response = await fetch(`${base}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url }),
        });
        if (!response.ok) {
            throw new Error(`predict failed: ${response.status}`);
        }
        const result = await response.json();
        let attribution = null;
        if (result.in_band && result.score !== null) {
            attribution = await fetchExplanation(base, url);
        }
        const { title, icon, message } = describe(result, attribution);
        chrome.notifications.create(`phishnet-${Date.now()}`, {
            type: "basic",
            requireInteraction: false,
            title: title,
            message: message,
            iconUrl: icon,
        });
    } catch (error) {
        console.error("PhishNet: check failed", error);
    }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.startsWith("http")) {
        checkUrl(tabId, tab.url);
    }
});

chrome.runtime.onInstalled.addListener(() => {
    getBackend().then((base) => {
        getHealth(base).then((health) => {
            if (health) console.log("PhishNet serving health", health);
        });
    });
});

console.log("PhishNet Detector (Phase 6) background loaded.");
