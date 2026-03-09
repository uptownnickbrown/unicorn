#!/usr/bin/env python3
"""
One-time setup: create a RunPod network volume and upload static data.

Creates a persistent volume, spins up a temporary pod to upload data files
that don't change between runs (parquet, embeddings, lookup CSVs), then
terminates the pod. The volume persists and can be attached to future pods
via --volume-id, eliminating the slow upload step.

Usage:
  python scripts/setup_volume.py                    # create volume + upload
  python scripts/setup_volume.py --volume-id vol_x  # upload to existing volume
  python scripts/setup_volume.py --list              # list existing volumes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_runpod import (
    PROJECT_DIR, CONTAINER_IMAGE,
    log, find_ssh_key, get_pod_ssh_info, ssh_exec, rsync_upload,
    wait_for_pod, wait_for_ssh, terminate_pod,
)

# Static data files that never change between runs — worth putting on volume
VOLUME_DATA_FILES = [
    "possessions.parquet",
    "base_player_text_embeddings.pt",
    "player_season_lookup.csv",
    "player_descriptions.jsonl",
    "requirements-train.txt",
]

VOLUME_NAME = "unicorn-data"
VOLUME_SIZE_GB = 10  # minimum RunPod volume size; we only use ~300MB
VOLUME_DIR = "/runpod-volume/unicorn"  # persistent volume mount point


def get_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Check .env or export it.")
        sys.exit(1)
    return api_key


def graphql_query(api_key: str, query: str, variables: dict | None = None) -> dict:
    """Execute a RunPod GraphQL query directly."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors'][0]['message']}")
    return data


def list_volumes(api_key: str) -> list[dict]:
    """List existing network volumes."""
    user = runpod.get_user()
    return user.get("networkVolumes", [])


def create_volume(api_key: str, name: str, size_gb: int, datacenter_id: str) -> str:
    """Create a network volume via GraphQL API."""
    query = """
    mutation createNetworkVolume($input: CreateNetworkVolumeInput!) {
        createNetworkVolume(input: $input) {
            id
            name
            size
            dataCenterId
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            "size": size_gb,
            "dataCenterId": datacenter_id,
        }
    }
    result = graphql_query(api_key, query, variables)
    vol = result["data"]["createNetworkVolume"]
    return vol["id"]


def save_volume_id(volume_id: str) -> None:
    """Append RUNPOD_VOLUME_ID to .env file."""
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        content = env_path.read_text()
        # Replace existing or append
        lines = content.splitlines()
        new_lines = [l for l in lines if not l.startswith("RUNPOD_VOLUME_ID=")]
        new_lines.append(f"RUNPOD_VOLUME_ID={volume_id}")
        env_path.write_text("\n".join(new_lines) + "\n")
    else:
        env_path.write_text(f"RUNPOD_VOLUME_ID={volume_id}\n")
    log(f"Saved RUNPOD_VOLUME_ID={volume_id} to .env")


def main():
    parser = argparse.ArgumentParser(description="Set up RunPod network volume with static data")
    parser.add_argument("--volume-id", default=None, help="Use existing volume (skip creation)")
    parser.add_argument("--list", action="store_true", help="List existing volumes and exit")
    parser.add_argument("--datacenter", default=None, help="Datacenter ID (e.g. US-TX-3)")
    parser.add_argument("--size", type=int, default=VOLUME_SIZE_GB, help=f"Volume size in GB (default: {VOLUME_SIZE_GB})")
    parser.add_argument("--gpu", default="NVIDIA RTX A5000", help="GPU for temp upload pod")
    args = parser.parse_args()

    api_key = get_api_key()
    runpod.api_key = api_key

    # List volumes
    if args.list:
        volumes = list_volumes(api_key)
        if not volumes:
            print("No network volumes found.")
        else:
            print(f"{'ID':<25s} {'Name':<20s} {'Size':>6s} {'Datacenter':<15s}")
            print("-" * 70)
            for v in volumes:
                print(f"{v['id']:<25s} {v['name']:<20s} {v['size']:>4d}GB {v['dataCenterId']:<15s}")
        return

    # Validate data files exist
    missing = [f for f in VOLUME_DATA_FILES if not (PROJECT_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing files: {', '.join(missing)}")
        sys.exit(1)

    ssh_key = find_ssh_key()
    volume_id = args.volume_id

    # Step 1: Create volume if needed
    if not volume_id:
        # Pick datacenter
        datacenter = args.datacenter
        if not datacenter:
            # List available datacenters from existing volumes or use default
            volumes = list_volumes(api_key)
            if volumes:
                # Use same datacenter as existing volumes
                datacenter = volumes[0]["dataCenterId"]
                log(f"Using datacenter from existing volume: {datacenter}")
            else:
                # Default to US datacenter
                datacenter = "US-TX-3"
                log(f"Using default datacenter: {datacenter}")

        log(f"Creating network volume '{VOLUME_NAME}' ({args.size}GB) in {datacenter}...")
        volume_id = create_volume(api_key, VOLUME_NAME, args.size, datacenter)
        log(f"Volume created: {volume_id}")
        save_volume_id(volume_id)
    else:
        log(f"Using existing volume: {volume_id}")
        # Save to .env if not already there
        save_volume_id(volume_id)

    # Step 2: Create temporary pod with volume attached
    log("\nCreating temporary pod for data upload...")
    pod = runpod.create_pod(
        name="unicorn-volume-setup",
        image_name=CONTAINER_IMAGE,
        gpu_type_id=args.gpu,
        network_volume_id=volume_id,
        container_disk_in_gb=5,
        ports="22/tcp",
        docker_args="",
    )
    pod_id = pod["id"]
    log(f"Temp pod created: {pod_id}")

    try:
        # Step 3: Wait for pod + SSH
        pod_info = wait_for_pod(pod_id)
        ssh_info = get_pod_ssh_info(pod_info)
        if not ssh_info:
            log("ERROR: Could not get SSH info")
            sys.exit(1)
        host, port = ssh_info
        wait_for_ssh(host, port, ssh_key)

        # Step 4: Create directory and upload
        ssh_exec(host, port, ssh_key, f"mkdir -p {VOLUME_DIR}", timeout=15)

        log(f"\nUploading {len(VOLUME_DATA_FILES)} static data files...")
        total_size = sum((PROJECT_DIR / f).stat().st_size for f in VOLUME_DATA_FILES)
        log(f"Total upload size: {total_size / (1024*1024):.0f} MB")

        rsync_upload(host, port, ssh_key, VOLUME_DATA_FILES, VOLUME_DIR)

        # Step 5: Verify
        log("\nVerifying uploaded files...")
        result = ssh_exec(host, port, ssh_key, f"ls -lh {VOLUME_DIR}/", timeout=15)
        print(result.stdout)

        log("Volume setup complete!")
        log(f"\nVolume ID: {volume_id}")
        log(f"Use with: python scripts/deploy_runpod.py --volume-id {volume_id}")
        log(f"      or: python scripts/deploy_ablation.py --volume-id {volume_id}")

    finally:
        # Always terminate the temp pod
        terminate_pod(pod_id)


if __name__ == "__main__":
    main()
