#!/usr/bin/env python3
"""汇总两个引擎的矩阵结果 → 对比表"""
import json
from pathlib import Path

D = Path("/tmp/llmbench")
A = json.loads((D / "ninfer.json").read_text())
B = json.loads((D / "llama.json").read_text())

CASES = ["short_decode", "mid", "long", "xl"]
LABEL = {"short_decode": "短(19 tok prompt)", "mid": "中(~10K)", "long": "长(~89K)", "xl": "超长(~123K)"}


def find(rows, case):
    for r in rows:
        if r.get("case") == case:
            return r
    return {}


def ttft(rows, idx=1):
    return find(rows, f"short_ttft#{idx}").get("ttft_ms")


print("| 档位 | 引擎 | prompt tok | 输出 tok | decode tok/s(客户端) | decode tok/s(服务端) | prefill tok/s |")
print("|---|---|---:|---:|---:|---:|---:|")
for c in CASES:
    for tag, rows in (("ninfer", A), ("llama.cpp", B)):
        r = find(rows, c)
        if not r or "error" in r:
            print(f"| {LABEL[c]} | {tag} | — | — | 失败 | — | — |")
            continue
        print(f"| {LABEL[c]} | {tag} | {r.get('prompt_tokens')} | {r.get('completion_tokens')} | "
              f"{r.get('tok_s')} | {r.get('srv_decode') or '—'} | {r.get('srv_prefill') or '—'} |")

print()
print("| 首字延迟(TTFT) | ninfer | llama.cpp |")
print("|---|---:|---:|")
for i, name in ((1, "冷(第1次)"), (2, "第2次"), (3, "第3次")):
    print(f"| {name} | {ttft(A, i)} ms | {ttft(B, i)} ms |")
