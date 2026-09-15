#!/usr/bin/env bash
# 自转 .ninfer 制品（groupwise-int 路线，用自选 bf16 checkpoint）
#
# 用法：
#   systemctl --user stop ninfer-serve     # 转换要独占 GPU，先腾显存
#   ./convert-abliterated.sh
#   （换源：MODEL=/path/to/other OUT=/path/out.ninfer ./convert-abliterated.sh）
#
# 前置：
#   - 源目录根部必须有 4 个 frontend 文件（tokenizer_config.json / generation_config.json /
#     preprocessor_config.json / video_preprocessor_config.json），且 sha256 与官方 Qwen/Qwen3.8-27B 一致
#     （缺件从官方 repo 拷入即可；converter 会逐字节校验）
#   - config.json 需与官方同构（层数/维度必须一致，微调只改权重值不影响）
#   - DFlash2 draft 模型目录（含 model.safetensors）
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/dale/comfy/ComfyUI/.venv/bin/python}"   # 有 torch + safetensors 的环境
MODEL="${MODEL:-/home/dale/models/huihui-qwen3.8-27b-abliterated}"
DF2="${DF2:-/home/dale/models/dflash2-qwen3.8-27b}"
OUT="${OUT:-/home/dale/models/ninfer/qwen3_8_27b_abliterated.ninfer}"
DEV="${DEV:-cuda}"

echo "python:   $PY"
echo "源模型:   $MODEL"
echo "DFlash2:  $DF2"
echo "输出:     $OUT"
echo "设备:     $DEV"
echo
if systemctl --user is-active --quiet ninfer-serve 2>/dev/null; then
  echo "⚠️  ninfer-serve 仍在运行（占着 ~27 GB 显存）。先执行： systemctl --user stop ninfer-serve"
  exit 1
fi
echo "✔ GPU 已腾空，开始转换（27B bf16 → 约 19 GiB 制品，逐张量流式，预计 30–90 分钟）"
echo

"$PY" -m tools.convert.qwen3_8_27b.convert \
  --model "$MODEL" \
  --dflash2-model "$DF2" \
  --out "$OUT" \
  --device "$DEV"
