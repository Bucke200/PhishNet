// PhishNet Detector - Phase 6 background service worker.
//
// Talks to the Phase 6 serving container (`phishnet.serving.app`): Tier-1
// score, fail-closed disposition, and native LightGBM SHAP. Thresholds are
// read from `/health` — never hard-coded here. The feedback endpoint
// (`/report`) was removed with the feedback pipeline; this worker has no
// write path.

const BACKEND_URL = "http://localhost:8000";
const PREDICT_ENDPOINT = `${BACKEND_URL}/predict`;
const EXPLAIN_ENDPOINT = `${BACKEND_URL}/explain`;
const HEALTH_ENDPOINT = `${BACKEND_URL}/health`;

let HEALTH = null;

async function getHealth() {
    if (HEALTH) return HEALTH;
    try {
        const response = await fetch(HEALTH_ENDPOINT);
        if (response.ok) HEALTH = await response.json();
    } catch (error) {
        console.warn("PhishNet: /health unavailable", error);
    }
    return HEALTH;
}

function formatScore(score) {
    return (score === null || score === undefined) ? "n/a" : Number(score).toFixed(4);
}

async function fetchExplanation(url) {
    try {
        const response = await fetch(EXPLAIN_ENDPOINT, {
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
        title = "Can't assess";
        icon = "icons/icon-safe.png";
    }

    const lines = [
        `${result.url}`,
        `disposition: ${disposition} (${result.reason})`,
        `Tier-1 score: ${formatScore(result.tier1_score)}`,
    ];
    if (disposition === "allow") {
        lines.push("Below the calibrated alert band - not a safety guarantee.");
    }
    if (result.tier2_mode && result.tier2_mode !== "disabled") {
        lines.push(`Tier 2 (${result.tier2_mode}): ${result.tier2 ? result.tier2.kind : "not run"}`);
    }
    if (attribution && attribution.features && attribution.features.length) {
        lines.push("Top features:");
        for (const f of attribution.features) {
            lines.push(`  ${f.feature}: ${Number(f.contribution).toFixed(3)}`);
        }
    }
    return { title, icon, message: lines.join("\n") };
}

async function checkUrl(tabId, url) {
    if (!url || !url.startsWith("http")) return;
    try {
        const response = await fetch(PREDICT_ENDPOINT, {
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
            attribution = await fetchExplanation(url);
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
    getHealth().then((health) => {
        if (health) console.log("PhishNet serving health", health);
    });
});

console.log("PhishNet Detector (Phase 6) background loaded.");
