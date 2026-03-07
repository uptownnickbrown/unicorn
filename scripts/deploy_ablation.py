#!/usr/bin/env python3
"""
Parallel RunPod ablation runner for Unicorn v4 attention experiments.

Creates 3 pods simultaneously, uploads data, runs training with different
attention configurations, polls all 3, downloads results, terminates pods.

Runs:
  A: Temperature-scaled attention (--attn-temperature 0.1)
  B: Entropy penalty (--attn-entropy-weight 0.1)
  C: Control (v3.2 config, no attention changes)

Usage:
  python scripts/deploy_ablation.py --epochs 5
  python scripts/deploy_ablation.py --epochs 5 --yes  # auto-terminate
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod")
    sys.exit(1)

# Reuse helpers from deploy_runpod.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_runpod import (
    PROJECT_DIR, CONTAINER_IMAGE, CONTAINER_DISK_GB, REMOTE_DIR,
    UPLOAD_FILES, GPU_FALLBACK_ORDER,
    log, find_ssh_key, get_pod_ssh_info, ssh_exec, rsync_upload,
    scp_download, create_pod, wait_for_pod, wait_for_ssh, terminate_pod,
    resolve_gpu,
)

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_RUNS = {
    "temp": {
        "name": "unicorn-ablation-temp",
        "ckpt": "ablation_temp_checkpoint",
        "extra_args": "--attn-temperature 0.1",
        "description": "Temperature-scaled attention (tau=0.1)",
    },
    "entropy": {
        "name": "unicorn-ablation-entropy",
        "ckpt": "ablation_entropy_checkpoint",
        "extra_args": "--attn-entropy-weight 0.1",
        "description": "Entropy penalty (lambda=0.1)",
    },
    "control": {
        "name": "unicorn-ablation-control",
        "ckpt": "ablation_control_checkpoint",
        "extra_args": "",
        "description": "Control (v3.2 config)",
    },
}

BASE_TRAIN_CMD = (
    "cd {remote_dir} && "
    "pip install -q -r requirements-train.txt && "
    "python train_transformer.py "
    "--phase joint --epochs {epochs} --delta-dim 64 "
    "--outcome-weight 1.0 --contrastive-weight 0.5 "
    "--prior-strength 10 --bs 2048 "
    "--ckpt {ckpt}.pt "
    "{extra_args}"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run 3 parallel ablation experiments on RunPod")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run (default: 5)")
    parser.add_argument("--gpu", default="NVIDIA RTX A5000", help="GPU type (default: A5000)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-terminate pods on completion")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--runs", nargs="+", default=list(ABLATION_RUNS.keys()),
                        choices=list(ABLATION_RUNS.keys()),
                        help="Which runs to execute (default: all)")
    args = parser.parse_args()

    # Validate environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Check .env or export it.")
        sys.exit(1)
    runpod.api_key = api_key

    gpu_type = resolve_gpu(args.gpu)
    ssh_key = find_ssh_key()

    runs = {k: ABLATION_RUNS[k] for k in args.runs}

    # Validate upload files
    missing = [f for f in UPLOAD_FILES if not (PROJECT_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing files: {', '.join(missing)}")
        sys.exit(1)

    # Show plan
    log("=" * 60)
    log("ABLATION PLAN")
    log("=" * 60)
    for key, run in runs.items():
        cmd = BASE_TRAIN_CMD.format(
            remote_dir=REMOTE_DIR, epochs=args.epochs,
            ckpt=run["ckpt"], extra_args=run["extra_args"],
        ).strip()
        log(f"  {key}: {run['description']}")
        log(f"    cmd: ...{run['extra_args'] or '(no extra args)'}")
    log(f"  GPU: {gpu_type}")
    log(f"  Epochs: {args.epochs}")
    log(f"  Estimated cost: ${len(runs) * args.epochs * 14/60 * 0.27:.2f}")
    log(f"  Estimated wall time: ~{args.epochs * 14:.0f} min")

    if args.dry_run:
        log("DRY RUN - exiting.")
        return

    start_time = time.time()
    pod_states = {}  # key -> {pod_id, host, port, done, success}

    # Safety net: terminate all pods on unexpected exit (crash, kill, etc.)
    def atexit_cleanup():
        for key, state in pod_states.items():
            pid = state.get("pod_id")
            if pid:
                log(f"  atexit: terminating {key} pod {pid}...")
                try:
                    terminate_pod(pid)
                except Exception:
                    log(f"  WARNING: failed to terminate {key} pod {pid}")
                    log(f"  Run: python scripts/runpod_cleanup.py --terminate")

    atexit.register(atexit_cleanup)

    try:
        # Step 1: Create all pods
        log("\n--- Creating pods ---")
        for key, run in runs.items():
            pod_id = create_pod(gpu_type, run["name"])
            pod_states[key] = {"pod_id": pod_id, "done": False, "success": False}

        # Step 2: Wait for all pods and get SSH info
        log("\n--- Waiting for pods ---")
        for key, state in pod_states.items():
            log(f"Waiting for {key} (pod {state['pod_id']})...")
            pod = wait_for_pod(state["pod_id"])
            ssh_info = get_pod_ssh_info(pod)
            if not ssh_info:
                log(f"ERROR: Could not get SSH info for {key}")
                continue
            state["host"], state["port"] = ssh_info
            wait_for_ssh(state["host"], state["port"], ssh_key)

        # Step 3: Upload data to all pods
        log("\n--- Uploading data ---")
        for key, state in pod_states.items():
            if "host" not in state:
                continue
            log(f"Uploading to {key}...")
            ssh_exec(state["host"], state["port"], ssh_key, f"mkdir -p {REMOTE_DIR}", timeout=15)
            rsync_upload(state["host"], state["port"], ssh_key, UPLOAD_FILES, REMOTE_DIR)

        # Step 4: Start training on all pods
        log("\n--- Starting training ---")
        for key, state in pod_states.items():
            if "host" not in state:
                continue
            run = runs[key]
            train_cmd = BASE_TRAIN_CMD.format(
                remote_dir=REMOTE_DIR, epochs=args.epochs,
                ckpt=run["ckpt"], extra_args=run["extra_args"],
            ).strip()
            setup_cmd = (
                f"tmux new-session -d -s train "
                f"'{train_cmd} 2>&1 | tee {REMOTE_DIR}/training.log'"
            )
            log(f"Starting {key}: {run['description']}")
            ssh_exec(state["host"], state["port"], ssh_key, setup_cmd, timeout=30)

        # Step 5: Poll all pods
        log(f"\n--- Monitoring (polling every {args.poll_interval}s) ---")
        while not all(s["done"] for s in pod_states.values()):
            time.sleep(args.poll_interval)

            for key, state in pod_states.items():
                if state["done"] or "host" not in state:
                    continue

                result = ssh_exec(
                    state["host"], state["port"], ssh_key,
                    "tmux has-session -t train 2>/dev/null && echo RUNNING || echo DONE",
                    check=False, timeout=15,
                )
                if result.returncode != 0:
                    continue

                # Show last log line
                tail_result = ssh_exec(
                    state["host"], state["port"], ssh_key,
                    f"tail -1 {REMOTE_DIR}/training.log 2>/dev/null || echo '(no log)'",
                    check=False, timeout=15,
                )
                if tail_result.returncode == 0 and tail_result.stdout.strip():
                    last = tail_result.stdout.strip()[:120]
                    print(f"  [{key}] {last}", flush=True)

                if result.stdout.strip() == "DONE":
                    run = runs[key]
                    ckpt_check = ssh_exec(
                        state["host"], state["port"], ssh_key,
                        f"test -f {REMOTE_DIR}/{run['ckpt']}_latest.pt && echo EXISTS || echo MISSING",
                        check=False, timeout=15,
                    )
                    state["done"] = True
                    state["success"] = ckpt_check.returncode == 0 and "EXISTS" in ckpt_check.stdout
                    status = "OK" if state["success"] else "FAILED"
                    log(f"  {key} finished: {status}")

        # Step 6: Download results from all pods
        log("\n--- Downloading results ---")
        for key, state in pod_states.items():
            if "host" not in state:
                continue
            run = runs[key]
            download_files = [
                f"{run['ckpt']}.pt",
                f"{run['ckpt']}.log.jsonl",
                f"{run['ckpt']}_latest.pt",
            ]
            for filename in download_files:
                remote_path = f"{REMOTE_DIR}/{filename}"
                local_path = str(PROJECT_DIR / filename)
                if scp_download(state["host"], state["port"], ssh_key, remote_path, local_path):
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    log(f"  {key}: {filename} ({size_mb:.1f} MB)")
                else:
                    log(f"  {key}: {filename} - not found")

            # Download training log
            log_remote = f"{REMOTE_DIR}/training.log"
            log_local = str(PROJECT_DIR / f"ablation_{key}_runpod.log")
            scp_download(state["host"], state["port"], ssh_key, log_remote, log_local)

    finally:
        # Step 7: Terminate all pods (default: terminate to prevent orphans)
        log("\n--- Cleanup ---")
        terminated = []
        for key, state in pod_states.items():
            if "pod_id" in state:
                if args.yes:
                    terminate_pod(state["pod_id"])
                    terminated.append(key)
                else:
                    resp = input(f"Keep {key} pod {state['pod_id']} running? [y/N] ")
                    if resp.strip().lower() == "y":
                        log(f"  {key} pod {state['pod_id']} left running!")
                    else:
                        terminate_pod(state["pod_id"])
                        terminated.append(key)
        # Clear terminated pods so atexit doesn't double-terminate
        for key in terminated:
            del pod_states[key]

    # Summary
    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log("ABLATION COMPLETE")
    log("=" * 60)
    log(f"  Wall time: {elapsed/60:.1f} minutes")
    for key, state in pod_states.items():
        run = runs[key]
        status = "OK" if state.get("success") else "FAILED"
        log(f"  {key} ({run['description']}): {status}")
    log("")
    log("Next: compare results with")
    log("  python -c \"import json; [print(json.loads(l)) for l in open('ablation_X_checkpoint.log.jsonl')]\"")
    log("  (replace X with temp, entropy, control)")


if __name__ == "__main__":
    main()
