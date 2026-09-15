# NInfer 在 Tesla V100 上的部署与实测

本目录是 NInfer 在 **Tesla V100-SXM2-32GB（Volta / sm_70）** 上把 **Qwen3.8-27B**
（自选 abliterated checkpoint）跑成 OpenAI/Anthropic 兼容 HTTP 服务的完整部署记录：
构建脚本、模型转换、systemd 托管、显存切换工具，以及跨引擎基准实测。

> 上游引擎仓库：`geoffwatts/ninfer-v100`。本目录是个人部署层（build + 模型转换 + 服务 + 基准），
> 不含引擎源码与权重（权重见 HuggingFace 各模型卡）。

## 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA Tesla V100-SXM2-32GB（Volta, SM 70） |
| 显存 | 32768 MiB |
| 驱动 | 580.173.02 |
| 系统 | Linux Mint 22.3（Cinnamon / Xorg） |
| 端口 | `127.0.0.1:8080`（OpenAI Chat/Responses + Anthropic Messages） |

## 实测效果（V100，Qwen3.8-27B，KV int8，MTP K=3，max-context 131072）

跨引擎对比（同脚本 `bench_matrix.py`、同语料、同档位；客户端视角 + 服务端上报）：

| 档位 | 引擎 | prompt tok | 输出 tok | decode tok/s（客户端） | decode tok/s（服务端） | prefill tok/s |
|---|---|---:|---:|---:|---:|---:|
| 短（19 tok prompt） | ninfer | 19 | 28 | **60.1** | **78.3** | 59.3 |
| 短（19 tok prompt） | llama.cpp | 19 | 28 | 37.8 | 52.3 | 18.0 |
| 中（~10K） | ninfer | 9989 | 82 | 8.1 | **62.2** | **1137.5** |
| 中（~10K） | llama.cpp | 9989 | 108 | 7.7 | 43.2 | 863.8 |
| 长（~89K） | ninfer | 89734 | 94 | 0.68 | **44.2** | **657.9** |
| 长（~89K） | llama.cpp | 89734 | 95 | 0.53 | 20.1 | 467.3 |
| 超长（~123K） | ninfer | 122658 | 101 | 0.43 | **32.4** | **534.3** |
| 超长（~123K） | llama.cpp | 122658 | 141 | 1.18 | 21.3 | 300.5 |

| 首字延迟 TTFT | ninfer | llama.cpp |
|---|---:|---:|
| 冷（第 1 次） | **299 ms** | 3171 ms |
| 第 2 次 | **120 ms** | 483 ms |
| 第 3 次 | **125 ms** | 221 ms |

要点：
- **decode 服务端吞吐** 在短/中/长/超长各档均显著领先 llama.cpp（短档 78 vs 52；超长 32 vs 21 tok/s）。
- **TTFT 热态约 120 ms**，比 llama.cpp 快 ~3–4 倍（前缀复用 + INT8 KV）。
- 服务日志实时上报 MTP 接受率 35–68%、decode 53–69 tok/s（视上下文长度与思考档位）。
- `tok/s（客户端）` 含端到端 wall 时间（短输出时 TTFT 占比大，数字偏低属正常）。
- 原始数据：`bench/results/ninfer.json`、`bench/results/llama.json`；复算表 `bench/summary.py`。

## 模型：两条自转换路线

权重不入库，源 checkpoint 自选（hf 拉取的 abliterated bf16）。两条路线都能出 `.ninfer` 制品：

1. **groupwise-int**（`convert-abliterated.sh`）
   - bf16 → 约 16.9 GiB 制品，逐张量流式，V100 上 30–90 分钟。
   - 需要 DFlash2 draft 模型目录（MTP 投机用）。
2. **NVFP4（W4A4）混合量化**（`quantize-nvfp4.py` / `quantize-nvfp4-manual.py`）
   - 先用 `llmcompressor` 做 compressed-tensors 混合量化源（FP8 组 + NVFP4 组），
     再由 NInfer `convert_nvfp4` 转成 `.ninfer`（约 23.7 GiB）。
   - targets 正则必须与 converter 的 `_FP8_TARGETS`/`_NVFP4_TARGETS` 逐字一致，否则 config 校验不过。
   - `input_global_scale` 直接复用官方 nvfp4 制品里提取的每层 divisor（同架构微调，免重校准）。

> 两条路线的 targets/常量都标注了与上游 converter 对齐的位置，改架构时同步改。

## 目录与脚本

```
deploy-v100/
├── README.md                 # 本文件（部署 + 实测）
├── scripts/
│   ├── build-v100.sh         # 构建 sm70 引擎二进制 + bench + serve
│   ├── convert-abliterated.sh# 自转 groupwise-int .ninfer
│   ├── quantize-nvfp4.py     # llmcompressor 混合量化源（FP8+NVFP4）
│   ├── quantize-nvfp4-manual.py # 手写分片量化（显存友好）
│   ├── k-sweep.sh            # MTP --draft-tokens K 值扫描
│   ├── bench-v100.sh         # 官方口径基准（prefill / decode / no-spec 对照）
│   ├── ninfer-serve.service  # systemd --user 单元（已脱敏为模板）
│   ├── wait-gpu.sh           # 启动前 GPU 驱动就绪探测（防 CPU 回退）
│   └── gpu-mode              # LLM / llama / ComfyUI 显存切换
└── bench/
    ├── bench_matrix.py       # 跨引擎对比矩阵（短/中/长/超长 + TTFT）
    ├── mkcorpus.py           # 生成本地测试语料
    ├── summary.py            # 汇总两引擎结果为对比表
    └── results/              # 实测 JSON（ninfer.json / llama.json）
```

## 部署步骤

### 1. 构建引擎

```bash
./scripts/build-v100.sh          # 产出 build-v100/apps/ninfer-serve、build-v100/bench/ninfer_bench
```

### 2. 准备模型（二选一）

```bash
systemctl --user stop ninfer-serve     # 转换要独占 GPU，先腾显存
./scripts/convert-abliterated.sh        # groupwise-int 路线
# 或 NVFP4：先跑 quantize-nvfp4.py 出量化源，再走 converter
```

产出后建软链，serve 永远读 `~/models/ninfer/current.ninfer`：

```bash
ln -sfn ~/models/ninfer/<your>.ninfer ~/models/ninfer/current.ninfer
```

### 3. systemd 托管

模板见 `scripts/ninfer-serve.service`（脱敏后路径/模型 ID 按本机改）。核心启动参数：

```bash
ninfer-serve ~/models/ninfer/current.ninfer \
  --host 127.0.0.1 --port 8080 \
  --model-id qwen3.8-27b \
  --max-context 131072 --kv-capacity auto \
  --prefill-chunk 2048 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --vision --max-concurrency 1
```

- `ExecStartPre=wait-gpu.sh`：开机时若 GPU 驱动未就绪会回退 CPU，此探测阻塞直到驱动可用，配合 `Restart=on-failure` 自愈。
- `--spec mtp --draft-tokens 3`：MTP 投机解码，K 值用 `k-sweep.sh` 扫。
- `--kv-dtype int8` + `--max-context 131072`：131K 上下文的 KV 放得进 32G。
- `--vision`：开图像/视频输入。
- `--max-concurrency 1`：单机单模型场景串行最稳。

```bash
systemctl --user enable --now ninfer-serve
journalctl --user -u ninfer-serve -f    # 看 req# 行：decode tok/s、mtp accepted
```

### 4. 显存切换（与 ComfyUI / llama.cpp 互斥）

V100 显存被 ninfer、llama.cpp、ComfyUI 三方共享，不能同时跑满。`gpu-mode` 统一调度：

```bash
gpu-mode llm      # 停 ComfyUI + 旧 llama，起 ninfer（现役主力）
gpu-mode llama    # 回退旧 llama.cpp
gpu-mode comfy    # 停 LLM，起 ComfyUI（出图/视频）
gpu-mode status   # 服务 + 显存总览
```

坑：`ninfer-serve` 没开 `SO_REUSEADDR`，llama 刚停时 8080 残留占用，直接 start 会
`cannot bind`。`gpu-mode` 用 socket bind 探测等到端口真空闲再起。

## 复现基准

```bash
./scripts/bench-v100.sh                # 官方口径（需先腾显存：停 llama-server）
# 跨引擎对比：
BENCH_TAG=ninfer python3 bench/bench_matrix.py   # ninfer
# 切到 llama-server 后：
BENCH_TAG=llama python3 bench/bench_matrix.py
python3 bench/summary.py                  # 出对比表
```

> 磊哥偏好：性能实测先清场——停掉争抢资源的任务（如死循环 dsh）再测，不接受被干扰的数字。
