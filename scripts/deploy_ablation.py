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
    PROJECT_DIR, CONTAINER_IMAGE, CONTAINER_DISK_GB, REMOTE_DIR, VOLUME_DIR,
    UPLOAD_FILES, CODE_FILES, GPU_FALLBACK_ORDER,
    log, find_ssh_key, get_pod_ssh_info, ssh_exec, rsync_upload,
    scp_download, create_pod, wait_for_pod, wait_for_ssh, terminate_pod,
    resolve_gpu,
)

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_RUNS = {
    "crossattn": {
        "name": "unicorn-v5-crossattn",
        "ckpt": "ablation_v5a_checkpoint",
        "extra_args": "--pool-type cross-attn --pool-heads 4",
        "description": "A: Cross-attention pooling (state-conditioned query)",
    },
    "multilayer": {
        "name": "unicorn-v5-multilayer",
        "ckpt": "ablation_v5b_checkpoint",
        "extra_args": "--pool-type cross-attn --pool-heads 4 --pool-multi-layer",
        "description": "B: Cross-attn + multi-layer pooling input",
    },
    "film": {
        "name": "unicorn-v5-film",
        "ckpt": "ablation_v5c_checkpoint",
        "extra_args": "--pool-type cross-attn --pool-heads 4 --pool-multi-layer --film-state",
        "description": "C: Cross-attn + multi-layer + FiLM state conditioning",
    },
}

BASE_TRAIN_CMD = (
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
    "--ckpt {ckpt}.pt "
    "{extra_args}"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run 3 parallel ablation experiments on RunPod")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run (default: 5)")
    parser.add_argument("--gpu", default="NVIDIA RTX A5000", help="GPU type (default: A5000, confirmed CUDA 12.9 compat)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-terminate pods on completion")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--runs", nargs="+", default=list(ABLATION_RUNS.keys()),
                        choices=list(ABLATION_RUNS.keys()),
                        help="Which runs to execute (default: all)")
    parser.add_argument("--volume-id", default=None,
                        help="Network volume ID (skips static data upload)")
    parser.add_argument("--no-volume", action="store_true",
                        help="Skip volume, upload all data (access any datacenter)")
    args = parser.parse_args()

    # Validate environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Check .env or export it.")
        sys.exit(1)
    runpod.api_key = api_key

    # Auto-detect volume ID from .env (unless --no-volume)
    if args.no_volume:
        args.volume_id = None
    elif not args.volume_id:
        args.volume_id = os.environ.get("RUNPOD_VOLUME_ID")

    gpu_type = resolve_gpu(args.gpu)
    ssh_key = find_ssh_key()

    runs = {k: ABLATION_RUNS[k] for k in args.runs}

    # Determine upload files (code only if volume attached)
    upload_files = CODE_FILES if args.volume_id else UPLOAD_FILES
    if args.volume_id:
        log(f"Using network volume {args.volume_id} — uploading code only")

    # Validate upload files
    missing = [f for f in upload_files if not (PROJECT_DIR / f).exists()]
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
            pod_id = create_pod(gpu_type, run["name"], args.volume_id)
            pod_states[key] = {"pod_id": pod_id, "done": False, "success": False}

        # Step 2: Wait for all pods and get SSH info (skip pods that fail)
        log("\n--- Waiting for pods ---")
        for key, state in pod_states.items():
            log(f"Waiting for {key} (pod {state['pod_id']})...")
            try:
                pod = wait_for_pod(state["pod_id"])
                ssh_info = get_pod_ssh_info(pod)
                if not ssh_info:
                    log(f"WARNING: Could not get SSH info for {key} — skipping")
                    state["done"] = True
                    continue
                state["host"], state["port"] = ssh_info
                wait_for_ssh(state["host"], state["port"], ssh_key)
            except (TimeoutError, Exception) as e:
                log(f"WARNING: {key} pod failed to start: {e} — skipping")
                state["done"] = True
                # Terminate the bad pod immediately
                try:
                    terminate_pod(state["pod_id"])
                except Exception:
                    pass

        # Step 3: Upload data to all pods
        log("\n--- Uploading data ---")
        for key, state in pod_states.items():
            if "host" not in state:
                continue
            log(f"Uploading to {key}...")
            ssh_exec(state["host"], state["port"], ssh_key, f"mkdir -p {REMOTE_DIR}", timeout=15)
            if args.volume_id:
                ssh_exec(state["host"], state["port"], ssh_key,
                         f"cp {VOLUME_DIR}/* {REMOTE_DIR}/ 2>/dev/null || true",
                         timeout=60)
            rsync_upload(state["host"], state["port"], ssh_key, upload_files, REMOTE_DIR)

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

                # Show last epoch summary (not tqdm progress bars)
                tail_result = ssh_exec(
                    state["host"], state["port"], ssh_key,
                    f"grep -E 'temporal.*top100|New best' {REMOTE_DIR}/training.log 2>/dev/null | tail -1",
                    check=False, timeout=15,
                )
                if tail_result.returncode == 0 and tail_result.stdout.strip():
                    epoch_result = ssh_exec(
                        state["host"], state["port"], ssh_key,
                        f"grep -c 'temporal.*top100' {REMOTE_DIR}/training.log 2>/dev/null",
                        check=False, timeout=15,
                    )
                    ep = epoch_result.stdout.strip() if epoch_result.returncode == 0 else "?"
                    last = tail_result.stdout.strip()[:120]
                    print(f"  [{key}] [ep {ep}/{args.epochs}] {last}", flush=True)

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
                    if not state["success"]:
                        # Show last 20 lines of training log for diagnosis
                        err_log = ssh_exec(
                            state["host"], state["port"], ssh_key,
                            f"tail -20 {REMOTE_DIR}/training.log 2>/dev/null || echo 'No training.log found'",
                            check=False, timeout=15,
                        )
                        log(f"  {key} error log:\n{err_log.stdout.strip()}")

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
