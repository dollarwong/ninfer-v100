#!/usr/bin/env python3
"""手写分片量化：huihui bf16 → compressed-tensors 混合量化源（供 ninfer convert_nvfp4）

- FP8 组（weight + weight_scale，per-channel BF16）：self_attn.(q|k|v|o)_proj /
  linear_attn.(in_proj_qkv|in_proj_z|out_proj) / lm_head / layers.56-63 的 MLP
- NVFP4 组（weight_packed U8 + weight_scale F8_E4M3 + weight_global_scale F32 +
  input_global_scale F32）：其余 MLP（layers 0-55）
- input_global_scale 直接复用官方 nvfp4 制品里提取的每层 divisor（同架构微调）

显存友好：逐张量处理，输出按 ~2 GB 切 shard
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/dale/repos/ninfer-v100")
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8  # noqa: E402
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA  # noqa: E402

SRC = Path("/home/dale/models/huihui-qwen3.8-27b-abliterated")
DST = Path("/home/dale/models/huihui-nvfp4-fp8")
DIVISOR_JSON = Path("/home/dale/models/official-input-divisors.json")
DEV = torch.device("cuda")
SHARD_BYTES = 2 * 1024**3
NVFP4_MAX_LAYER = 55          # 56-63 的 MLP 归 FP8 组

FP8_PATS = [
    r".*self_attn\.(q|k|v|o)_proj$",
    r".*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r".*lm_head$",
    r".*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
]
NVFP4_PATS = [r".*mlp\.(gate|up|down)_proj$"]


def group_of(mod: str) -> str | None:
    if any(re.match(p, mod) for p in FP8_PATS):
        return "fp8"
    if any(re.match(p, mod) for p in NVFP4_PATS):
        return "nvfp4"
    return None


def layer_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def fp8_quant(w: torch.Tensor):
    wf = w.to(DEV).float()
    scale = (wf.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    q = (wf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.cpu(), scale.to(torch.bfloat16).cpu()


def nvfp4_quant(w: torch.Tensor, wgs: float | None = None):
    wf = w.to(DEV).float()
    n, k = wf.shape
    if wgs is None:
        wgs = (448.0 * 6.0 / wf.abs().max().clamp(min=1e-12)).item()
    ws = wf * wgs
    blk = ws.view(n, k // 16, 16)
    bmax = blk.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    bscale = (bmax / 6.0).to(torch.float8_e4m3fn)          # (n, k/16, 1)
    q = (blk / bscale.float()).view(n, k)
    q = FP4_E2M1_DATA.cast_to_fp4(q.contiguous())
    packed = pack_fp4_to_uint8(q)                          # (n, k//2) uint8
    return (packed.cpu(), bscale.view(n, k // 16).cpu(),
            torch.tensor([wgs], dtype=torch.float32))


def main() -> None:
    index = json.load(open(SRC / "model.safetensors.index.json"))
    weight_map: dict[str, str] = index["weight_map"]
    divs = json.load(open(DIVISOR_JSON))

    targets = {}
    for name in weight_map:
        if not name.endswith(".weight"):
            continue
        if name.startswith("mtp."):
            continue          # MTP 层不量化（converter 的 FP8/NVFP4 源都不含 mtp）
        mod = name[:-len(".weight")]
        g = group_of(mod)
        if g is None:
            continue
        if g == "nvfp4":
            lay = layer_of(name)
            if lay is None or lay > NVFP4_MAX_LAYER:
                continue      # 56-63 已由 FP8 规则覆盖；无层的（异常）跳过
        targets[name] = g

    n_fp8 = sum(1 for g in targets.values() if g == "fp8")
    n_nvfp4 = sum(1 for g in targets.values() if g == "nvfp4")
    print(f"待量化：FP8 {n_fp8} 个 + NVFP4 {n_nvfp4} 个 = {len(targets)} 个", flush=True)

    DST.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, torch.Tensor]] = {}
    weight_out: dict[str, str] = {}
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    shard_idx = 0

    def flush():
        nonlocal cur, cur_bytes, shard_idx
        if not cur:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(cur, DST / fname, metadata={"format": "pt"})
        for k in cur:
            weight_out[k] = fname
        print(f"  [shard {shard_idx}] {fname}: {len(cur)} 张量", flush=True)
        cur, cur_bytes = {}, 0

    # 逐 shard / 逐张量读源
    readers = {}
    fused_amax: dict[str, float] = {}
    done = 0
    for name, g in targets.items():
        shard = weight_map[name]
        if shard not in readers:
            readers[shard] = safe_open(SRC / shard, framework="pt")
        w = readers[shard].get_tensor(name)
        mod = name[:-len(".weight")]
        if g == "fp8":
            q, s = fp8_quant(w)
            cur[mod + ".weight"] = q
            cur[mod + ".weight_scale"] = s
            cur_bytes += q.numel() + s.numel() * 2
        else:
            lay = layer_of(name)
            kind = "gate_up" if ("gate_proj" in mod or "up_proj" in mod) else "down"
            wgs_val = None
            if kind == "gate_up":
                mlp = mod.rsplit(".", 1)[0]
                if mlp not in fused_amax:
                    a = 0.0
                    for nm in (mlp + ".gate_proj.weight", mlp + ".up_proj.weight"):
                        sh = weight_map[nm]
                        if sh not in readers:
                            readers[sh] = safe_open(SRC / sh, framework="pt")
                        a = max(a, readers[sh].get_tensor(nm).float().abs().max().item())
                    fused_amax[mlp] = a
                wgs_val = 448.0 * 6.0 / max(fused_amax[mlp], 1e-12)
            packed, bscale, wgs = nvfp4_quant(w, wgs_val)
            key = f"text/layers/{lay}/mlp/{kind}_projection/input_scale_divisor"
            igs = divs.get(key)
            if igs is None:
                print(f"  ⚠️ 缺少 divisor {key}，用 1.0 占位", flush=True)
                igs = 1.0
            cur[mod + ".weight_packed"] = packed
            cur[mod + ".weight_scale"] = bscale
            cur[mod + ".weight_global_scale"] = wgs
            cur[mod + ".input_global_scale"] = torch.tensor([igs], dtype=torch.float32)
            cur_bytes += packed.numel() + bscale.numel() + 8
        done += 1
        if done % 25 == 0:
            print(f"  进度 {done}/{len(targets)}", flush=True)
        if cur_bytes >= SHARD_BYTES:
            flush()
    flush()

    # 非量化张量：复用 base 的 shard（symlink 零拷贝，converter 要求量化源是完整模型目录）
    reused = 0
    for name, shard in weight_map.items():
        if name not in targets:
            weight_out[name] = shard
            reused += 1
    for shard in set(weight_map.values()):
        link = DST / shard
        if not link.exists():
            link.symlink_to((SRC / shard).resolve())
    print(f"复用 base 张量 {reused} 个（symlink {len(set(weight_map.values()))} 个 shard）", flush=True)

    # index.json
    total = sum((DST / s).stat().st_size for s in set(weight_out.values()))
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": weight_out},
        open(DST / "model.safetensors.index.json", "w"), indent=1,
    )

    # config.json：复制 base 的 config + quantization_config
    cfg = json.load(open(SRC / "config.json"))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": [
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*lm_head",
                    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
                ],
                "weights": {"num_bits": 8, "type": "float", "symmetric": True,
                            "strategy": "channel", "group_size": None,
                            "dynamic": False, "scale_dtype": None},
                "input_activations": {"num_bits": 8, "type": "float", "symmetric": True,
                                      "strategy": "token", "group_size": None,
                                      "dynamic": True, "scale_dtype": None},
                "output_activations": None,
                "format": "float-quantized",
            },
            "group_1": {
                "targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
                "weights": {"num_bits": 4, "type": "float", "symmetric": True,
                            "strategy": "tensor_group", "group_size": 16,
                            "dynamic": False, "scale_dtype": "torch.float8_e4m3fn"},
                "input_activations": {"num_bits": 4, "type": "float", "symmetric": True,
                                      "strategy": "tensor_group", "group_size": 16,
                                      "dynamic": "local", "scale_dtype": "torch.float8_e4m3fn"},
                "output_activations": None,
                "format": "nvfp4-pack-quantized",
            },
        },
        "quantization_status": "compressed",
    }
    json.dump(cfg, open(DST / "config.json", "w"), indent=1)

    # 其余 frontend 文件（converter 要从量化源读 config.json 的 summary，只需 config + index + safetensors）
    print(f"\n完成：{len(weight_out)} 个量化张量，{len(set(weight_out.values()))} 个 shard → {DST}", flush=True)


if __name__ == "__main__":
    main()
