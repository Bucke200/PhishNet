"""Dispatch phase4-forward.yml via the API using the stored git credential.

Never prints the credential. Exits 1 with the manual command if no
credential is available non-interactively.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request

OWNER_REPO = ("Bucke200", "PhishNet")
WORKFLOW = "phase4-forward.yml"


def stored_token() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input=b"protocol=https\nhost=github.com\n\n",
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    user, pw = "", ""
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("username="):
            user = line[len("username="):]
        elif line.startswith("password="):
            pw = line[len("password="):]
    token = pw or (user if len(user) > 20 else "")
    return token or None


def main() -> int:
    token = stored_token()
    if not token:
        print("no non-interactive credential; run by hand:")
        print(f"  gh workflow run {WORKFLOW} --repo "
              f"{OWNER_REPO[0]}/{OWNER_REPO[1]} --ref master")
        return 1
    owner, repo = OWNER_REPO
    url = (f"https://api.github.com/repos/{owner}/{repo}"
           f"/actions/workflows/{WORKFLOW}/dispatches")
    body = json.dumps({"ref": "master"}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "PhishNet-dispatch"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"dispatch HTTP {resp.status}")
            return 0 if resp.status in (201, 204) else 1
    except Exception as exc:  # noqa: BLE001 — report, never leak headers
        print(f"dispatch failed: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
