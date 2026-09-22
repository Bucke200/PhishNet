"""Operator CLI for the LLM spend guard (`phishnet.llm.budget`).

Prints the cumulative ledger for the active budget id, clears a stale run
lock, or zeroes the ledger. Reads the same environment as the guard:

  PHISHNET_LLM_BUDGET_USD   cap for this budget id (unset = track only)
  PHISHNET_LLM_BUDGET_ID    ledger namespace (default "default")
  PHISHNET_LLM_BUDGET_DIR   ledger/lock directory (default ".budget")

Usage:
  uv run python scripts/llm_budget.py                 # status
  uv run python scripts/llm_budget.py --unlock        # clear "run" lock
  uv run python scripts/llm_budget.py --reset --yes   # zero the ledger
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from phishnet.llm import budget  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unlock",
        nargs="?",
        const="run",
        default=None,
        metavar="NAME",
        help="remove the named run lock (default: run)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="zero the ledger for the current budget id",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required with --reset; confirms the destructive action",
    )
    args = parser.parse_args(argv)

    if args.unlock is not None:
        path = budget.lock_path(str(args.unlock))
        budget.release_lock(str(args.unlock))
        print(f"released lock (if present): {path}")
        return 0

    if args.reset:
        if not args.yes:
            print("refusing --reset without --yes", file=sys.stderr)
            return 2
        budget.reset()
        print("ledger zeroed")
        return 0

    print(json.dumps(budget.status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
