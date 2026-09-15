#!/usr/bin/env bash
# 官方口径复现：与上游 README 表格同参数
#   prefill = 单独 pp2048；decode = pp2048+tg256；CUDA Graph；优化 proposal head；
#   INT8 KV；丢 1 次 warmup + 3 次实测
set -uo pipefail
cd /home/dale/repos/ninfer-v100

BENCH=./build-v100/bench/ninfer_bench
ART=/home/dale/models/ninfer/current.ninfer
OUT=/tmp/nvfp4-official-sweep.txt
: > "$OUT"

echo "### artifact: $(readlink -f $ART)" | tee -a "$OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$OUT"
echo | tee -a "$OUT"

echo "===== PREFILL 单独 pp2048 (int8 KV) =====" | tee -a "$OUT"
"$BENCH" --weights "$ART" -p 2048 --kv-dtype int8 --max-ctx 8192 \
         --warmup 1 -r 3 -o table 2>&1 | tee -a "$OUT"

for K in 1 2 3 4 5; do
  echo | tee -a "$OUT"
  echo "===== DECODE pp2048+tg256  MTP K=$K  (int8 KV, lm-head-draft) =====" | tee -a "$OUT"
  "$BENCH" --weights "$ART" -pg '2048,256' --spec mtp --draft-tokens "$K" --lm-head-draft \
           --kv-dtype int8 --max-ctx 8192 --warmup 1 -r 3 -o table 2>&1 | tee -a "$OUT"
done

echo | tee -a "$OUT"
echo "===== 参考：decode 无投机 pp2048+tg256 =====" | tee -a "$OUT"
"$BENCH" --weights "$ART" -pg '2048,256' --kv-dtype int8 --max-ctx 8192 \
         --warmup 1 -r 3 -o table 2>&1 | tee -a "$OUT"

echo | tee -a "$OUT"
echo "全部完成" | tee -a "$OUT"
