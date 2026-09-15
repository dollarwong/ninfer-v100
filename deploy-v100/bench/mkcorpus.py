#!/usr/bin/env python3
"""生成长/中/短测试语料（与历史测试同尺寸：40KB≈10K tok / 260KB≈85K / 390KB≈130K）"""
import random
from pathlib import Path

SRC_DIRS = [Path.home() / "wiki", Path.home() / "文档"]
SIZES = {"mid": 17_500, "long": 148_750, "xl": 205_000}  # 按 ~1.66 字符/token ⇒ 10K / 89K / 123K token
OUT = Path("/tmp/llmbench")
OUT.mkdir(exist_ok=True)

pool = []
for d in SRC_DIRS:
    if not d.exists():
        continue
    for f in d.rglob("*.md"):
        try:
            t = f.read_text(errors="ignore").strip()
        except Exception:
            continue
        if len(t) > 200:
            pool.append(t)

print(f"池子: {len(pool)} 个文档")
random.seed(42)
random.shuffle(pool)
joined = "\n\n".join(pool)

for name, size in SIZES.items():
    buf = []
    n = 0
    while n < size:
        buf.append(joined[n % len(joined):(n % len(joined)) + 4000])
        n = sum(len(x) for x in buf)
    text = "".join(buf)[:size]
    p = OUT / f"corpus-{name}.txt"
    p.write_text(text)
    print(f"  {p.name}: {len(text)} 字符")
