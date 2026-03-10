#!/usr/bin/env python3
"""
Automated RunPod GPU training for Unicorn v4.

Creates a pod, uploads data, runs training, downloads results, terminates pod.

Prerequisites:
  1. RunPod account with $25+ credit: https://runpod.io
  2. API key in .env (RUNPOD_API_KEY=...) or exported in shell
  3. SSH public key added to RunPod settings
  4. Install: pip install runpod

Usage:
  python scripts/deploy_runpod.py

  # Options:
  python scripts/deploy_runpod.py --gpu "NVIDIA RTX 4090"  # default
  python scripts/deploy_runpod.py --gpu "NVIDIA RTX A4000"  # cheaper
  python scripts/deploy_runpod.py --epochs 10               # fewer epochs
  python scripts/deploy_runpod.py --dry-run                  # show plan, don't execute
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (RUNPOD_API_KEY, etc.)
_env_file = PROJECT_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Static data files (large, never change — should live on network volume)
VOLUME_DATA_FILES = [
    "possessions.parquet",
    "base_player_text_embeddings.pt",
    "player_season_lookup.csv",
    "player_descriptions.jsonl",
    "requirements-train.txt",
]

# Code files (small, change between runs — always upload fresh)
CODE_FILES = [
    "nba_dataset.py",
    "train_transformer.py",
    "prior_year_init.py",
    "evaluate.py",
    "analyze_embeddings.py",
]

# All files (used when no volume attached)
UPLOAD_FILES = VOLUME_DATA_FILES + CODE_FILES

# Files to download after training
DOWNLOAD_FILES = [
    "joint_v4_checkpoint.pt",
    "joint_v4_checkpoint.log.jsonl",
    "joint_v4_checkpoint_latest.pt",
]

CONTAINER_IMAGE = "runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204"  # CUDA 12.9, needs driver >= 570
CONTAINER_DISK_GB = 20
REMOTE_DIR = "/workspace/unicorn"
VOLUME_DIR = "/runpod-volume/unicorn"  # persistent volume mount point

# Training command — CUDA check prevents silent CPU fallback (35s/batch vs 0.4s/batch)
# NOTE: No single quotes! This gets wrapped in tmux '...' which breaks on nested single quotes.
TRAIN_CMD_TEMPLATE = (
    "cd {remote_dir} && "
    "echo === CUDA CHECK === && "
    "python -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\" && "
    "echo === PIP INSTALL === && "
    "pip install -q -r requirements-train.txt && "
    "echo === TRAINING START === && "
    "python train_transformer.py "
    "--phase joint --epochs {epochs} --delta-dim 64 "
    "--outcome-weight 1.0 --contrastive-weight 0.5 "
    "--prior-strength 10 --bs 2048 "
    "--pool-type cross-attn --pool-heads 4 --pool-multi-layer --film-state "
    "--ckpt joint_v5_checkpoint.pt"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def find_ssh_key() -> str:
    """Find an SSH private key to use."""
    for name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
        path = Path.home() / ".ssh" / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "No SSH key found in ~/.ssh/. Add one to RunPod settings and retry."
    )


def get_pod_ssh_info(pod: dict) -> tuple[str, int] | None:
    """Extract SSH host and port from pod runtime info."""
    runtime = pod.get("runtime")
    if not runtime:
        return None
    ports = runtime.get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22:
            ip = p.get("ip")
            public_port = p.get("publicPort")
            if ip and public_port:
                return (ip, int(public_port))
    return None


def ssh_cmd(host: str, port: int, key: str) -> list[str]:
    """Build base SSH command list."""
    return [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-i", key,
        "-p", str(port),
        f"root@{host}",
    ]


def ssh_exec(host: str, port: int, key: str, command: str,
             check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Execute a command on the pod via SSH."""
    cmd = ssh_cmd(host, port, key) + [command]
    return subprocess.run(
        cmd, check=check, capture_output=True, text=True, timeout=timeout,
    )


def rsync_upload(host: str, port: int, key: str, files: list[str],
                 remote_dir: str) -> None:
    """Upload files to the pod via rsync."""
    # Build file list from project dir
    file_paths = []
    for f in files:
        p = PROJECT_DIR / f
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        file_paths.append(str(p))

    cmd = [
        "rsync", "-avz", "--progress", "--no-owner", "--no-group",
        "-e", f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i {key} -p {port}",
    ] + file_paths + [f"root@{host}:{remote_dir}/"]

    log(f"Uploading {len(files)} files...")
    subprocess.run(cmd, check=True)


def scp_download(host: str, port: int, key: str, remote_path: str,
                 local_path: str) -> bool:
    """Download a file from the pod via scp. Returns True if successful."""
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-i", key,
        "-P", str(port),
        f"root@{host}:{remote_path}",
        local_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

GPU_FALLBACK_ORDER = [
    "NVIDIA RTX A5000",          # $0.27/hr, confirmed CUDA 12.9 compatible
    "NVIDIA GeForce RTX 4090",   # $0.69/hr, fast but pricier
    "NVIDIA RTX A6000",          # $0.53/hr
    "NVIDIA L40",                # $0.69/hr
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA RTX A4000",          # $0.16/hr, slower
    "NVIDIA GeForce RTX 3090",   # $0.22/hr, some machines have old drivers — CUDA check catches this
]


def create_pod(gpu_type: str, pod_name: str, network_volume_id: str | None = None) -> str:
    """Create a RunPod pod, falling back to other GPUs if unavailable."""
    # Build candidate list: requested GPU first, then fallbacks
    candidates = [gpu_type] + [g for g in GPU_FALLBACK_ORDER if g != gpu_type]

    for gpu in candidates:
        log(f"Trying {gpu}...")
        try:
            kwargs = dict(
                name=pod_name,
                image_name=CONTAINER_IMAGE,
                gpu_type_id=gpu,
                cloud_type="ALL",  # search both secure + community cloud
                container_disk_in_gb=CONTAINER_DISK_GB,
                ports="22/tcp",
                docker_args="",
            )
            if network_volume_id:
                kwargs["network_volume_id"] = network_volume_id
            else:
                kwargs["volume_in_gb"] = 0
            pod = runpod.create_pod(**kwargs)
            pod_id = pod["id"]
            log(f"Pod created: {pod_id} ({gpu})")
            return pod_id
        except Exception as e:
            err = str(e)
            if "resources" in err.lower() or "not have" in err.lower() or "no longer any instances" in err.lower():
                log(f"  {gpu} unavailable, trying next...")
                continue
            raise

    raise RuntimeError("No GPUs available. Try again later.")


def wait_for_pod(pod_id: str, timeout_sec: int = 1200) -> dict:
    """Wait for pod to be running and SSH-ready (image pull can take 10-15 min)."""
    log("Waiting for pod to start...")
    start = time.time()
    while time.time() - start < timeout_sec:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime:
            ssh_info = get_pod_ssh_info(pod)
            if ssh_info:
                log(f"Pod running. SSH: {ssh_info[0]}:{ssh_info[1]}")
                return pod

        elapsed = int(time.time() - start)
        print(f"  Status: {status} ({elapsed}s elapsed)...", end="\r", flush=True)
        time.sleep(5)

    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout_sec}s")


def wait_for_ssh(host: str, port: int, key: str, timeout_sec: int = 120) -> None:
    """Wait until SSH is accepting connections."""
    log("Waiting for SSH to be ready...")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            result = ssh_exec(host, port, key, "echo ready", check=False, timeout=10)
            if result.returncode == 0:
                log("SSH connection established.")
                return
        except (subprocess.TimeoutExpired, Exception):
            pass
        time.sleep(5)
    raise TimeoutError(f"SSH not ready after {timeout_sec}s")


def start_training(host: str, port: int, key: str, epochs: int) -> None:
    """Start training in a tmux session on the pod."""
    train_cmd = TRAIN_CMD_TEMPLATE.format(remote_dir=REMOTE_DIR, epochs=epochs)

    # Use tmux so training survives SSH disconnects
    setup_cmd = (
        f"mkdir -p {REMOTE_DIR} && "
        f"tmux new-session -d -s train '{train_cmd} 2>&1 | tee {REMOTE_DIR}/training.log'"
    )
    log("Starting training in tmux session...")
    ssh_exec(host, port, key, setup_cmd, timeout=30)
    log("Training started.")


def poll_training(host: str, port: int, key: str, epochs: int,
                  poll_interval: int = 60) -> bool:
    """Poll training progress until completion or failure."""
    log(f"Monitoring training ({epochs} epochs, polling every {poll_interval}s)...")
    consecutive_failures = 0

    while True:
        time.sleep(poll_interval)

        # Check if tmux session still exists (training still running)
        result = ssh_exec(
            host, port, key,
            "tmux has-session -t train 2>/dev/null && echo RUNNING || echo DONE",
            check=False, timeout=15,
        )

        if result.returncode != 0:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log("WARNING: Lost SSH connection (5 consecutive failures)")
                return False
            continue
        consecutive_failures = 0

        status = result.stdout.strip()

        # Show latest epoch summary (grep for metrics, not tqdm progress bars)
        tail_result = ssh_exec(
            host, port, key,
            f"grep -E 'temporal.*top100|New best|outcome_loss' {REMOTE_DIR}/training.log 2>/dev/null | tail -2",
            check=False, timeout=15,
        )
        if tail_result.returncode == 0:
            lines = tail_result.stdout.strip()
            if lines:
                # Count completed epochs
                epoch_result = ssh_exec(
                    host, port, key,
                    f"grep -c 'temporal.*top100' {REMOTE_DIR}/training.log 2>/dev/null",
                    check=False, timeout=15,
                )
                epoch_count = epoch_result.stdout.strip() if epoch_result.returncode == 0 else "?"
                for line in lines.splitlines():
                    print(f"  [{time.strftime('%H:%M:%S')}] [ep {epoch_count}/{epochs}] {line.strip()}", flush=True)

        if status == "DONE":
            # Check if checkpoint exists (success indicator)
            ckpt_result = ssh_exec(
                host, port, key,
                f"test -f {REMOTE_DIR}/joint_v4_checkpoint.pt && echo EXISTS || echo MISSING",
                check=False, timeout=15,
            )
            if ckpt_result.returncode == 0 and "EXISTS" in ckpt_result.stdout:
                log("Training completed successfully!")
                return True
            else:
                log("Training process ended but no checkpoint found. Check logs.")
                # Download training log for debugging
                return False

    return False


def download_results(host: str, port: int, key: str) -> list[str]:
    """Download checkpoint and log files."""
    downloaded = []
    for filename in DOWNLOAD_FILES:
        remote_path = f"{REMOTE_DIR}/{filename}"
        local_path = str(PROJECT_DIR / filename)
        log(f"Downloading {filename}...")
        if scp_download(host, port, key, remote_path, local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            log(f"  {filename} ({size_mb:.1f} MB)")
            downloaded.append(filename)
        else:
            log(f"  {filename} - not found (skipping)")

    # Always try to download the training log
    log_remote = f"{REMOTE_DIR}/training.log"
    log_local = str(PROJECT_DIR / "training_runpod.log")
    if scp_download(host, port, key, log_remote, log_local):
        downloaded.append("training_runpod.log")

    return downloaded


def terminate_pod(pod_id: str) -> None:
    """Terminate the pod."""
    log(f"Terminating pod {pod_id}...")
    try:
        runpod.terminate_pod(pod_id)
        log("Pod terminated.")
    except Exception as e:
        log(f"WARNING: Failed to terminate pod {pod_id}: {e}")
        log(f"  Manually terminate at: https://runpod.io/console/pods")


# ---------------------------------------------------------------------------
# GPU type mapping
# ---------------------------------------------------------------------------

GPU_ALIASES = {
    "4090": "NVIDIA GeForce RTX 4090",
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4090": "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090": "NVIDIA GeForce RTX 4090",
    "a4000": "NVIDIA RTX A4000",
    "RTX A4000": "NVIDIA RTX A4000",
    "NVIDIA RTX A4000": "NVIDIA RTX A4000",
    "a5000": "NVIDIA RTX A5000",
    "RTX A5000": "NVIDIA RTX A5000",
    "NVIDIA RTX A5000": "NVIDIA RTX A5000",
    "3090": "NVIDIA GeForce RTX 3090",
    "RTX 3090": "NVIDIA GeForce RTX 3090",
    "a6000": "NVIDIA RTX A6000",
    "RTX A6000": "NVIDIA RTX A6000",
}


def resolve_gpu(name: str) -> str:
    return GPU_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deploy Unicorn v4 training to RunPod GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpu", default="NVIDIA GeForce RTX 4090",
        help="GPU type (default: RTX 4090). Aliases: 4090, a4000, 3090, a6000",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--pod-name", default="unicorn-v4", help="Pod name")
    parser.add_argument("--ssh-key", default=None, help="SSH private key path (auto-detected)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--skip-upload", action="store_true", help="Skip file upload (for resumed pods)")
    parser.add_argument("--pod-id", default=None, help="Attach to existing pod instead of creating one")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-terminate pod on completion (no prompt)")
    parser.add_argument("--volume-id", default=None,
                        help="Network volume ID (skips static data upload). Set up with: python scripts/setup_volume.py")
    parser.add_argument("--no-volume", action="store_true",
                        help="Skip volume, upload all data (access any datacenter)")
    args = parser.parse_args()

    # Auto-detect volume ID from .env if not specified (unless --no-volume)
    if args.no_volume:
        args.volume_id = None
    elif not args.volume_id:
        args.volume_id = os.environ.get("RUNPOD_VOLUME_ID")

    # Validate environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set.")
        print("  1. Get your key at https://runpod.io/console/user/settings")
        print("  2. export RUNPOD_API_KEY=your_key")
        sys.exit(1)

    runpod.api_key = api_key
    gpu_type = resolve_gpu(args.gpu)
    ssh_key = args.ssh_key or find_ssh_key()

    # Determine which files to upload
    if args.volume_id:
        upload_files = CODE_FILES  # static data already on volume
        log(f"Using network volume {args.volume_id} — uploading code only ({len(CODE_FILES)} files)")
    else:
        upload_files = UPLOAD_FILES

    # Validate upload files exist
    missing = [f for f in upload_files if not (PROJECT_DIR / f).exists()]
    if missing and not args.skip_upload:
        print(f"ERROR: Missing files: {', '.join(missing)}")
        print(f"  Run from project root: {PROJECT_DIR}")
        sys.exit(1)

    # Dry run
    if args.dry_run:
        log("DRY RUN - would execute:")
        log(f"  GPU: {gpu_type}")
        log(f"  Image: {CONTAINER_IMAGE}")
        log(f"  Epochs: {args.epochs}")
        log(f"  SSH key: {ssh_key}")
        log(f"  Upload: {len(UPLOAD_FILES)} files")
        total_size = sum((PROJECT_DIR / f).stat().st_size for f in UPLOAD_FILES if (PROJECT_DIR / f).exists())
        log(f"  Upload size: {total_size / (1024*1024):.0f} MB")
        log(f"  Train cmd: {TRAIN_CMD_TEMPLATE.format(remote_dir=REMOTE_DIR, epochs=args.epochs)}")
        return

    # Track pod ID for cleanup (mutable so closures see updates)
    state = {"pod_id": args.pod_id}

    def cleanup_handler():
        pid = state["pod_id"]
        if pid:
            log(f"\nAuto-terminating pod {pid} (atexit safety net)...")
            try:
                runpod.terminate_pod(pid)
                log("  Pod terminated.")
            except Exception as e:
                log(f"  WARNING: Failed to terminate pod {pid}: {e}")
                log(f"  Manually terminate: python scripts/runpod_cleanup.py --terminate")

    atexit.register(cleanup_handler)

    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        log("\nInterrupted! Attempting to download partial results...")
        pid = state["pod_id"]
        if pid:
            try:
                pod = runpod.get_pod(pid)
                ssh_info = get_pod_ssh_info(pod)
                if ssh_info:
                    download_results(ssh_info[0], ssh_info[1], ssh_key)
            except Exception as e:
                log(f"Could not download results: {e}")

            resp = input(f"\nTerminate pod {pid}? [y/N] ")
            if resp.strip().lower() == "y":
                terminate_pod(pid)
            else:
                log(f"Pod {pid} left running. Remember to terminate it!")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()

    # Step 1: Create or attach to pod
    if state["pod_id"]:
        log(f"Attaching to existing pod: {state['pod_id']}")
    else:
        state["pod_id"] = create_pod(gpu_type, args.pod_name, args.volume_id)

    pod_id = state["pod_id"]

    # Step 2: Wait for pod
    pod = wait_for_pod(pod_id)
    ssh_info = get_pod_ssh_info(pod)
    if not ssh_info:
        log("ERROR: Could not get SSH connection info from pod")
        sys.exit(1)
    host, port = ssh_info

    # Step 3: Wait for SSH
    wait_for_ssh(host, port, ssh_key)

    # Step 4: Create remote directory and seed from volume if available
    ssh_exec(host, port, ssh_key, f"mkdir -p {REMOTE_DIR}", timeout=15)
    if args.volume_id:
        log("Copying data from volume to workspace...")
        ssh_exec(host, port, ssh_key,
                 f"cp {VOLUME_DIR}/* {REMOTE_DIR}/ 2>/dev/null || true",
                 timeout=60)

    # Step 5: Upload data (code only if volume attached, everything otherwise)
    if not args.skip_upload:
        rsync_upload(host, port, ssh_key, upload_files, REMOTE_DIR)

    # Step 6: Start training
    start_training(host, port, ssh_key, args.epochs)

    # Step 7: Poll for completion
    success = poll_training(host, port, ssh_key, args.epochs, args.poll_interval)

    # Step 8: Download results
    downloaded = download_results(host, port, ssh_key)

    # Step 9: Terminate pod
    elapsed = time.time() - start_time
    elapsed_hrs = elapsed / 3600

    log("")
    log("=" * 60)
    log("TRAINING COMPLETE" if success else "TRAINING ENDED (check logs)")
    log("=" * 60)
    log(f"  Wall time:  {elapsed_hrs:.1f} hours")
    log(f"  Downloaded: {', '.join(downloaded) if downloaded else 'none'}")
    log("")

    if args.yes:
        terminate_pod(pod_id)
        state["pod_id"] = None
    else:
        resp = input(f"Terminate pod {pod_id}? [Y/n] ")
        if resp.strip().lower() != "n":
            terminate_pod(pod_id)
            state["pod_id"] = None
        else:
            log(f"Pod {pod_id} left running. Remember to terminate it!")
            log(f"  runpod pod terminate {pod_id}")

    # Verification hint
    if success and "joint_v4_checkpoint.pt" in downloaded:
        log("")
        log("Next steps:")
        log("  python fit_deltas.py --ckpt joint_v4_checkpoint_latest.pt")
        log("  python scripts/precompute_eval.py --ckpt joint_v4_checkpoint_latest.pt")


if __name__ == "__main__":
    main()
