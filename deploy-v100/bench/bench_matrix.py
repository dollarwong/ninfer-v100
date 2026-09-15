#!/usr/bin/env python3
"""跨引擎对比矩阵（同一 prompt / 同一脚本 / 同档位）：短·中·长·超长 × decode + TTFT"""
import json
import os
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-27b")
TAG = os.environ.get("BENCH_TAG", "engine")
EFFORT = os.environ.get("BENCH_EFFORT", "none")
OUTDIR = Path("/tmp/llmbench")
OUTDIR.mkdir(exist_ok=True)
RESULTS = []

SHORT_PROMPT = "用一句话回答：什么是光合作用？"
INSTR = "\n\n请根据以上材料，用 150 字左右总结其核心内容。"


def post(payload, timeout=1200, stream=False):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            return json.loads(r.read().decode()), time.time() - t0
        ttft = None
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                delta = json.loads(d)["choices"][0].get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
            except Exception:
                pass
        return ttft, time.time() - t0


def record(name, body, wall, extra=None):
    u = body.get("usage", {}) or {}
    t = body.get("timings", {}) or {}
    row = {
        "case": name,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "wall_s": round(wall, 2),
        "tok_s": round((u.get("completion_tokens") or 0) / wall, 2) if wall else None,
        "srv_prefill": round(t.get("prompt_per_second"), 1) if t.get("prompt_per_second") else None,
        "srv_decode": round(t.get("predicted_per_second"), 1) if t.get("predicted_per_second") else None,
    }
    if extra:
        row.update(extra)
    RESULTS.append(row)
    print(f"  {name}: prompt={row['prompt_tokens']} out={row['completion_tokens']} "
          f"wall={row['wall_s']}s => {row['tok_s']} tok/s | srv decode={row['srv_decode']}", flush=True)
    return row


def case_short():
    print("=== 短档（无上下文，固定小问题）===", flush=True)
    for i in range(3):
        ttft, wall = post({"model": MODEL, "messages": [{"role": "user", "content": SHORT_PROMPT}],
                           "temperature": 0, "reasoning_effort": EFFORT,
                           "max_tokens": 120, "stream": True}, stream=True)
        RESULTS.append({"case": f"short_ttft#{i + 1}", "ttft_ms": round((ttft or 0) * 1000)})
        print(f"  short TTFT #{i + 1}: {(ttft or 0) * 1000:.0f} ms", flush=True)
    body, wall = post({"model": MODEL, "messages": [{"role": "user", "content": SHORT_PROMPT}],
                       "temperature": 0, "reasoning_effort": EFFORT, "max_tokens": 160})
    record("short_decode", body, wall)


def case_corpus(name):
    text = (OUTDIR / f"corpus-{name}.txt").read_text()
    print(f"=== {name} 档（{len(text)} 字符语料）===", flush=True)
    body, wall = post({"model": MODEL,
                       "messages": [{"role": "user", "content": text + INSTR}],
                       "temperature": 0, "reasoning_effort": EFFORT, "max_tokens": 200})
    record(name, body, wall)


if __name__ == "__main__":
    case_short()
    for n in ("mid", "long", "xl"):
        if (OUTDIR / f"corpus-{n}.txt").exists():
            try:
                case_corpus(n)
            except Exception as e:
                print(f"  {n} 档失败: {str(e)[:150]}", flush=True)
                RESULTS.append({"case": n, "error": str(e)[:200]})
    out = OUTDIR / f"{TAG}.json"
    out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    print(f"\n结果写入 {out}", flush=True)
