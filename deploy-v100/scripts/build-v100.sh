#!/usr/bin/env bash
# ninfer-v100 sm_70 (Tesla V100 / Volta) 构建脚本
# 前置：CUDA Toolkit 12.8（不能是 13.x，13 已删 Volta 离线编译）、CMake>=3.28、Ninja、C++20 编译器、
#       ffmpeg 开发库（libavformat>=60 / libavcodec>=60 / libavutil>=58 / libswscale>=7）、libcurl>=7.85
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.8}"
if [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
  echo "ERROR: 未找到 $CUDA_ROOT/bin/nvcc —— 先装 CUDA 12.8 toolkit" >&2
  exit 1
fi
NVCC_VER="$("$CUDA_ROOT/bin/nvcc" --version | tail -1)"
echo "CUDA: $NVCC_VER   (要求 12.8 <= v < 13.0)"

cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_BUILD_BENCHMARKS=ON

# 并行度：默认取 min(nproc, 6)，可用 NINFER_JOBS 覆盖（nvcc 每个进程吃 1-3 GB 内存，别贪多）
NPROC="$(nproc)"
JOBS="${NINFER_JOBS:-$(( NPROC < 6 ? NPROC : 6 ))}"
echo "并行编译任务数: $JOBS (nproc=$NPROC)"
cmake --build build-v100 -j"$JOBS"

echo
echo "=== 构建完成 ==="
ls -la build-v100/bench/ninfer_bench build-v100/apps/ninfer build-v100/apps/ninfer-serve 2>/dev/null || true
