# NInfer

> Up to 219 decode tok/s from Qwen 3.8 27B on a single V100.  With software NVFP4 on Volta.

NInfer is a from-scratch C++/CUDA inference engine optimized for selected Qwen checkpoints on NVIDIA Tesla V100.

It supports text, image, and video input through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs. The runtime is intentionally narrow: one GPU, one resident model, 1–8 active requests.

> **Fork notes** — this fork targets a real single-card deployment and adds:
> - `perf(sm70)`: accelerated sm_70 vision encoding (upstream `Neroued/ninfer`).
> - **[`deploy-v100/`](deploy-v100/README.md)** — end-to-end deployment of Qwen3.8-27B (abliterated) on a Tesla V100-SXM2-32GB: build/convert scripts, systemd unit, `wait-gpu` boot gate, `gpu-mode` VRAM switching, and a cross-engine benchmark (NInfer vs llama.cpp) with measured results.

## Models

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Artifacts contain the exact model weights, tokenizer, chat template, and required media frontend resources.

## Performance

Qwen3.8-27B NVFP4 reaches **218.98 decode tok/s** at K=1, with 99.2% MTP draft acceptance.
That result is on a V100-PCIe-32GB, not SXM. The equivalent SXM2 card is roughly 7% faster; decode is predominantly HBM-bound, so PCIe bandwidth and host performance have little effect.

### Tesla V100: software NVFP4 and groupwise inference

The Qwen3.8-27B NVFP4 short-context target round, retested after the width-6+ verify fix below,
peaks at K=1 draft tokens: **219.0 decode tok/s** at 99.2% draft acceptance -- narrow windows win
outright on this corpus; see the full K sweep below.

The single-request sweep uses the public Engine benchmark on a Tesla V100-PCIe-32GB with CUDA
12.8 and INT8 group-64 KV. Prefill is an isolated `pp2048` run; decode is `pp2048+tg256` with CUDA
Graphs and the optimized proposal head. Each result uses one discarded warmup and three measured
repetitions. On the DFlash window sweep in the V100 notes the preferred V100-SXM2-32GB ran about
7% faster per round; this decode workload is HBM-bound, so the host and PCIe bus barely matter.

| Model profile | K | Prefill tok/s | Decode tok/s | Draft acceptance |
|---|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` MTP | 4 | 1,085.0 | 54.54 | 66.5% |
| Qwen3.6-27B `nvfp4` MTP | 5 | 223.8 | 55.22 | 54.4% |
| Qwen3.8-27B `groupwise-int` MTP | 5 | 1,083.9 | 130.96 | 97.1% |
| Qwen3.8-27B `nvfp4` MTP | 5 | 1,100.3 | 199.58 | 97.1% |
| Qwen3.8-27B `groupwise-int` DFlash2 | 7 | 1,044.2 | 77.84 | 100% |
| Qwen3.8-27B `nvfp4` DFlash2 | 7 | 1,059.0 | 126.32 | 100% |
| Qwen3.6-35B-A3B `groupwise-int` DFlash | 4 | 686.2 | 139.58 | 90.9% |


Full Qwen3.8-27B `nvfp4` MTP draft-window sweep on this same corpus-continuation shape, now that
the width-6+ fix removes the sm_70 cap at four:

| K | Prefill tok/s | Decode tok/s | Draft acceptance |
|---:|---:|---:|---:|
| 1 | 1,102.5 | **218.98** | 99.2% |
| 2 | 1,097.4 | 213.99 | 98.3% |
| 3 | 1,094.9 | 209.24 | 97.5% |
| 4 | 1,096.6 | 204.02 | 97.9% |
| 5 | 1,100.3 | 199.58 | 97.1% |
| 6 | 1,089.6 | 178.47 | 92.5% |
| 7 | 1,087.2 | 180.74 | 93.3% |

This corpus is a deterministic continuation with unusually high, near-ceiling acceptance at every
K, so narrow windows win outright: round-verify cost dominates once there's little more accepted
length to buy. Treat these as a synthetic-corpus ceiling, not a general-generation rate -- on real,
less predictable text the practical production sweet spot is K=3 (see the long-context sweep
elsewhere in this repo's history).

Decode throughput depends strongly on draft acceptance -- see the MTP sweep above for how much.
The sm_70 width-6+ target-verify regression is fixed (see below), so `--spec mtp` now accepts the
same [1,7] window upstream does, no Volta-specific cap. `--spec dflash2` peaks at K=7 on this same
corpus-continuation shape (a 3-10 sweep falls off on both sides); MTP still leads DFlash2 here at
every K tried.

The Qwen3.8-27B artifacts also bundle DFlash2, the upstream masked-block speculative decoder
(`--spec dflash2`). MTP remains the recommended Volta backend for general decoding. On a varied,
non-repetitive corpus, DFlash2 K=7 narrowly beat the best MTP window at 2K and 32K context, while
MTP led at 8K, 16K, and 150K. DFlash2 K=7 is its strongest static default; acceptance-driven
window adaptation can favor K=3 on difficult continuations. See the full comparison in the
[V100 port notes](docs/v100.md#varied-context-dflash2-sweep).

MTP automatically extends verification up to fifteen draft tokens when the generated suffix
exactly matches an earlier 16-token span and the learned proposal agrees with the lookup
continuation. On a 172-token verbatim-copy prompt, Qwen3.8-27B NVFP4 produced the exact
continuation at **201.0 tok/s**, averaging 12.91 output tokens per round. This is a
context-reproduction fast path; ordinary generation continues to use the normal MTP window and
the general decode results above.

Context-lookup MTP was inspired by
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090).

On Volta, NVFP4 is decoded and executed in software using tuned FP16 tensor-core and SIMT kernels.
Dense MLP gate/up payloads are prepacked in place during model load for the QPN decode layout; the
artifact on disk is unchanged and inference does not perform runtime weight repacking. The Volta
QPN prepacking work was inspired by
[dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny).

The 35B-A3B production DFlash round at a 2,048-token context uses K=3: **125.9 tok/s** and 3.8
mean output tokens per round over ten measured rounds after two warmups.

## Quick start

NInfer requires 64-bit Linux, a Tesla V100 with CUDA Toolkit 12.8, CMake 3.28 or newer, a C++20
host compiler, Ninja, `pkg-config`, FFmpeg
development libraries (`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and
`libswscale >= 7`), and `libcurl >= 7.85`. This port builds for `sm_70`.

Select CUDA 12.8 and Volta explicitly:

```bash
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

The same five `.ninfer` artifacts use the public Engine/CLI/serving routes. See the
[V100 port notes](docs/v100.md) for qualification and the preferred-GPU launcher.

On Volta, loading an NVFP4 artifact repacks each dense MLP gate/up and down payload in place on the
GPU into the QPN fragment order used by the decode kernels. Row-scaled FP8 payloads are likewise
permuted into their native Volta QPN stream. The `.ninfer` file and host-side artifact bytes
are unchanged, the device allocation remains the same size, and temporary repack storage is
released before inference. These device-resident weights are therefore deliberately mutated at
load; the V100 path does not retain checkpoint-native immutable layout for those payloads.

Tests, benchmarks, and maintainer tools are excluded from the default build. There is no install
target or packaged binary distribution; run NInfer from its source build tree.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/vision agent server with one active-request lane and Device/Host
checkpoint retention:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --prefill-chunk 2048 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

Each request has a 240,000-token logical ceiling. The Device KV pool is sized from the memory left
after weights, workspace, state, Vision, and graph allocations. The cache tiers provide one Device
checkpoint slot, eight pinned Host State slots, and 8 GiB of pinned Host KV.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Structured startup/runtime-error records and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization additionally uses one transient progress line. Redirected stderr receives only
persistent structured phase records, including rate-limited progress for long loads. Option and
local input errors remain direct command diagnostics. Use `--messages FILE` and `--vision` for
structured image/video input; see the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens for
its measured memory envelope. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution,
and DFlash2 for Qwen3.8-27B.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to seven;
- DFlash2 masked-block speculative decoding for Qwen3.8-27B (from the upstream integration),
  draft windows from one to fifteen;
- BF16, INT8, and FP8 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one Tesla V100 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [V100 qualification and performance](docs/v100.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
