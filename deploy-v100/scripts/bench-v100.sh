#!/usr/bin/env bash
# ninfer-v100 基准（官方 V100 口径）
#   官方条件：prefill = 单独 pp2048；decode = pp2048+tg256，CUDA Graph，optimized proposal head，
#             INT8 group-64 KV，1 次丢弃 warmup + 3 次实测，V100-PCIe-32GB。
# 前置：先腾显存 —— systemctl --user stop llama-server   （llama 占 22 GB，和本基准不能共存）
set -euo pipefail
cd "$(dirname "$0")"

BENCH=build-v100/bench/ninfer_bench
ART="${1:-/home/dale/models/ninfer/current.ninfer}"
OUT="${2:-/tmp/ninfer-bench-$(date +%H%M%S).txt}"

[ -x "$BENCH" ] || { echo "缺少 $BENCH，先跑 build-v100.sh" >&2; exit 1; }
[ -f "$ART" ] || { echo "缺少制品 $ART" >&2; exit 1; }

echo "=== 显存现状 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo

echo "=== 1) prefill 单独测 (pp2048) ==="
"$BENCH" --weights "$ART" -p 2048 --kv-dtype int8 --max-ctx 8192 \
         --warmup 1 -r 3 -o table | tee -a "$OUT"
echo

echo "=== 2) decode 测 (pp2048+tg256, MTP K=3) ==="
"$BENCH" --weights "$ART" -pg '2048,256' --spec mtp --draft-tokens 3 --lm-head-draft \
         --kv-dtype int8 --max-ctx 8192 --warmup 1 -r 3 -o table | tee -a "$OUT"
echo

echo "=== 3) decode 无投机对照 (pp2048+tg256, no spec) ==="
"$BENCH" --weights "$ART" -pg '2048,256' --kv-dtype int8 --max-ctx 8192 \
         --warmup 1 -r 3 -o table | tee -a "$OUT"
echo
echo "结果已写入: $OUT"
