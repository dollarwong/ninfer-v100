#!/usr/bin/env bash
# wait-gpu.sh — blocking pre-start gate for llama-server.
# Polls until the NVIDIA driver exposes a usable device to the CUDA runtime.
# Exits 0 (GPU ready) or 1 (timeout) so systemd Restart=on-failure retries.
#
# This fixes the boot-time race where llama-server started before the GPU
# driver was ready, silently fell back to CPU, and stayed there forever
# ("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device").
#
# FIX (2026-09-09): some nvidia-smi builds print the failure message
# ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA
# driver...") to STDOUT, not just stderr. The old code took that line as the
# GPU name via `head -1`, judged it non-empty, and declared the GPU ready at
# 0s — letting llama-server through while the driver was still down, which
# silently fell back to CPU. Now we check the failure marker explicitly.
set -u

MAX_WAIT="${GPU_WAIT_SECS:-90}"
step=2
elapsed=0

while [ "$elapsed" -lt "$MAX_WAIT" ]; do
  # Capture BOTH streams; nvidia-smi may emit the failure text on stdout.
  raw="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)"
  # Reject the failure marker and empty output (no usable GPU found).
  if [ -n "$raw" ] && ! printf '%s\n' "$raw" | grep -qiE \
      "has failed|couldn't communicate|no devices|cannot communicate|not supported|failed"; then
    echo "GPU ready after ${elapsed}s: ${raw}"
    exit 0
  fi
  sleep "$step"
  elapsed=$((elapsed + step))
done

echo "wait-gpu: no GPU appeared within ${MAX_WAIT}s (last check: nvidia-smi failed)" >&2
exit 1
