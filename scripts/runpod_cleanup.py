#!/usr/bin/env python3
"""
RunPod pod cleanup utility.

Lists all active pods and optionally terminates them. Use as a safety net
to catch orphaned pods that weren't terminated after training.

Usage:
  python scripts/runpod_cleanup.py              # list all pods
  python scripts/runpod_cleanup.py --terminate   # terminate all pods (with confirmation)
  python scripts/runpod_cleanup.py --terminate --yes  # terminate all without confirmation
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod")
    sys.exit(1)

# Load .env
PROJECT_DIR = Path(__file__).resolve().parent.parent
_env_file = PROJECT_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def main():
    parser = argparse.ArgumentParser(description="List and clean up RunPod pods")
    parser.add_argument("--terminate", action="store_true",
                        help="Terminate all listed pods")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Check .env or export it.")
        sys.exit(1)
    runpod.api_key = api_key

    pods = runpod.get_pods()
    if not pods:
        print("No active pods found.")
        return

    print(f"{'ID':<28} {'Name':<30} {'Status':<12} {'GPU':<20}")
    print("-" * 90)
    for p in pods:
        gpu = ""
        rt = p.get("runtime", {})
        if rt and rt.get("gpus"):
            gpu = rt["gpus"][0].get("id", "?")
        print(f"{p['id']:<28} {p['name']:<30} {p['desiredStatus']:<12} {gpu:<20}")

    print(f"\n{len(pods)} pod(s) found.")

    if not args.terminate:
        print("\nTo terminate all: python scripts/runpod_cleanup.py --terminate")
        return

    if not args.yes:
        resp = input(f"\nTerminate all {len(pods)} pod(s)? [y/N] ")
        if resp.strip().lower() != "y":
            print("Cancelled.")
            return

    for p in pods:
        try:
            runpod.terminate_pod(p["id"])
            print(f"  Terminated: {p['name']} ({p['id']})")
        except Exception as e:
            print(f"  FAILED to terminate {p['name']}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
