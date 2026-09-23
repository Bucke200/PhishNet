// Options page: configure the serving backend URL.
//
// The URL is stored in chrome.storage.local and read by the service worker.
// Because the extension's static host_permissions cover only localhost, a
// custom origin is requested here (from the Save click, which is a user
// gesture) via optional_host_permissions. After saving, /health is probed
// so a typo'd Cloud Run URL fails loudly here instead of silently later.

const DEFAULT_BACKEND =
    (typeof PHISHNET_DEFAULT_BACKEND !== "undefined" && PHISHNET_DEFAULT_BACKEND) ||
    "http://localhost:8000";

const input = document.getElementById("backend");
const status = document.getElementById("status");

function setStatus(text, isError) {
    status.textContent = text;
    status.className = isError ? "err" : "ok";
}

function normalizeBase(value) {
    const url = new URL(value.trim());
    const path = url.pathname.replace(/\/+$/, "");
    return `${url.origin}${path === "/" ? "" : path}`;
}

async function testConnection(base) {
    const response = await fetch(`${base}/health`);
    if (!response.ok) {
        throw new Error(`/health answered ${response.status}`);
    }
    const health = await response.json();
    if (health.status !== "ok") {
        throw new Error(`unhealthy backend: ${JSON.stringify(health)}`);
    }
    const hash = typeof health.model_hash === "string" ? health.model_hash.slice(0, 12) : "?";
    const tier2 = health.tier2_mode || "unknown";
    return `Connected: model ${hash}…, Tier-2 ${tier2}.`;
}

async function load() {
    const { backendUrl } = await chrome.storage.local.get("backendUrl");
    input.value = backendUrl || DEFAULT_BACKEND;
}

document.getElementById("save").addEventListener("click", async () => {
    let base;
    try {
        base = normalizeBase(input.value);
    } catch (error) {
        setStatus("Enter a valid URL, e.g. http://localhost:8000", true);
        return;
    }
    const origin = `${new URL(base).origin}/*`;
    try {
        const granted = await chrome.permissions.request({ origins: [origin] });
        if (!granted) {
            setStatus(`Permission for ${origin} was not granted.`, true);
            return;
        }
        await chrome.storage.local.set({ backendUrl: base });
        setStatus(`Saved: ${base} — testing…`, false);
        try {
            setStatus(`Saved: ${base} — ${await testConnection(base)}`, false);
        } catch (error) {
            setStatus(`Saved: ${base} — test failed: ${error.message}`, true);
        }
    } catch (error) {
        setStatus(`Could not save: ${error.message}`, true);
    }
});

load();
