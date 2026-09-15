#!/usr/bin/env python3
"""按 ninfer-v100 convert_nvfp4 的要求，把自选 bf16 checkpoint 量化为
compressed-tensors 混合方案：
  group_0  FP8_DYNAMIC  : self_attn.(q|k|v|o)_proj / linear_attn.(in_proj_qkv|in_proj_z|out_proj)
                          / lm_head / layers.56-63.mlp.(gate|up|down)_proj
  group_1  NVFP4 (W4A4) : 其余 mlp.(gate|up|down)_proj
targets 正则必须与 converter 的 _FP8_TARGETS / _NVFP4_TARGETS 逐字一致（否则 config 校验不过）。

用法：
  systemctl --user stop ninfer-serve     # 要独占显存
  /home/dale/venvs/nvfp4-quant/bin/python quantize-nvfp4.py
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL = os.environ.get("MODEL", "/home/dale/models/huihui-qwen3.8-27b-abliterated")
OUT = os.environ.get("OUT", "/home/dale/models/huihui-nvfp4-fp8")
CALIB = os.environ.get("CALIB", "/home/dale/models/nvfp4-calib-mixed.jsonl")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "256"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))

# 与 tools/convert/qwen3_8_27b/convert_nvfp4.py 中的常量逐字对齐
FP8_TARGETS = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
]
NVFP4_TARGETS = [r"re:.*mlp\.(gate|up|down)_proj$"]

recipe = [
    # 先做 FP8（含 56-63 层 MLP），再做 NVFP4 —— 顺序保证后者跳过已量化模块
    QuantizationModifier(targets=FP8_TARGETS, scheme="FP8_DYNAMIC"),
    QuantizationModifier(targets=NVFP4_TARGETS, scheme="NVFP4"),
]

print(f"加载模型 {MODEL}（device_map=auto，跨 GPU+CPU）...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print(f"加载校准集 {CALIB} ...", flush=True)
ds = load_dataset("json", data_files=CALIB, split="train")

print(f"开始量化：{N_SAMPLES} 样本 / 最长 {MAX_LEN} token → {OUT}", flush=True)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=N_SAMPLES,
    max_seq_length=MAX_LEN,
    output_dir=OUT,
)
print("量化完成 →", OUT, flush=True)
