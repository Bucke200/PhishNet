// Options page: configure the serving backend URL.
//
// The URL is stored in chrome.storage.local and read by the service worker.
// Because the extension's static host_permissions cover only localhost, a
// custom origin is requested here (from the Save click, which is a user
// gesture) via optional_host_permissions.

const DEFAULT_BACKEND = "http://localhost:8000";

const input = document.getElementById("backend");
const status = document.getElementById("status");

function setStatus(text, isError) {
    status.textContent = text;
    status.className = isError ? "err" : "ok";
}

async function load() {
    const { backendUrl } = await chrome.storage.local.get("backendUrl");
    input.value = backendUrl || DEFAULT_BACKEND;
}

document.getElementById("save").addEventListener("click", async () => {
    let url;
    try {
        url = new URL(input.value.trim());
    } catch (error) {
        setStatus("Enter a valid URL, e.g. http://localhost:8000", true);
        return;
    }
    const base = `${url.origin}${url.pathname.replace(/\/+$/, "")}`.replace(/\/+$/, "");
    const origin = `${url.origin}/*`;
    try {
        const granted = await chrome.permissions.request({ origins: [origin] });
        if (!granted) {
            setStatus(`Permission for ${origin} was not granted.`, true);
            return;
        }
        await chrome.storage.local.set({ backendUrl: base });
        setStatus(`Saved: ${base}`, false);
    } catch (error) {
        setStatus(`Could not save: ${error.message}`, true);
    }
});

load();
