"""Poll the dispatched phase4-forward run, then fetch its manifest.

Read-only over the API except no writes at all; never prints the credential.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

OWNER, REPO, WORKFLOW = "Bucke200", "PhishNet", "phase4-forward.yml"


def token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input=b"protocol=https\nhost=github.com\n\n",
        capture_output=True,
        timeout=60,
    )
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("password=") and line[len("password=") :]:
            return line[len("password=") :]
    raise SystemExit("no credential")


def api(path: str, tok: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{OWNER}/{REPO}{path}",
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "PhishNet-poll",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    return json.loads(raw)


def main() -> int:
    tok = token()
    deadline = time.time() + 30 * 60
    run_id = None
    while time.time() < deadline:
        runs = api(f"/actions/workflows/{WORKFLOW}/runs?per_page=3", tok)
        runs = runs.get("workflow_runs", [])
        if runs:
            run = runs[0]
            run_id = run["id"]
            print(
                f"run {run_id}: status={run['status']} "
                f"conclusion={run.get('conclusion')}",
                flush=True,
            )
            if run["status"] == "completed":
                print(
                    json.dumps(
                        {
                            "id": run_id,
                            "conclusion": run.get("conclusion"),
                            "html_url": run.get("html_url"),
                            "head_branch": run.get("head_branch"),
                            "head_sha": run.get("head_sha"),
                        },
                        indent=2,
                    )
                )
                return 0
        time.sleep(45)
    print("timed out waiting")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
