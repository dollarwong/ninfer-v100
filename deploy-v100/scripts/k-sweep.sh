#!/usr/bin/env bash
# ninfer 参数扫描：找出真实中文负载下最快的组合
# 每个组合独立跑一次 CLI（加载 ~13s + 生成 ~10s），stderr 全量留档
cd /home/dale/repos/ninfer-v100 || exit 1
BIN=./build-v100/apps/ninfer
ART=/home/dale/models/ninfer/current.ninfer
P='请用大约400字介绍郑州的交通枢纽地位，分点论述。'
OUT=/tmp/ninfer-sweep.txt
: > "$OUT"

run() {
  local tag="$1"; shift
  echo "===== $tag =====" >> "$OUT"
  "$BIN" "$ART" --prompt "$P" --max-context 8192 --max-new 400 \
    --reasoning-effort low "$@" >>"$OUT" 2>&1
  echo "" >> "$OUT"
}

run "A: mtp K=1 int8"  --kv-dtype int8 --spec mtp --draft-tokens 1 --lm-head-draft
run "B: mtp K=2 int8"  --kv-dtype int8 --spec mtp --draft-tokens 2 --lm-head-draft
run "C: mtp K=3 int8"  --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
run "D: mtp K=4 int8"  --kv-dtype int8 --spec mtp --draft-tokens 4 --lm-head-draft
run "E: mtp K=5 int8"  --kv-dtype int8 --spec mtp --draft-tokens 5 --lm-head-draft
run "F: dflash2 K=7 int8" --kv-dtype int8 --spec dflash2 --draft-tokens 7 --lm-head-draft
run "G: mtp K=3 fp8"   --kv-dtype fp8  --spec mtp --draft-tokens 3 --lm-head-draft
run "H: mtp K=3 int8 prefill-chunk 2048" --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 2048
echo "全部完成" >> "$OUT"
