#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#SBATCH -J SimVP-Folsom
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:8            # 通过 GRES 分配 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128GB
#SBATCH -t 72:00:00
#SBATCH -o /mnt/nfs/yuan/slurm_logs/%x-%j.out
#SBATCH -e /mnt/nfs/yuan/slurm_logs/%x-%j.err

# ===== 环境 =====
eval "$(micromamba shell hook --shell bash)"
micromamba activate SimVP

# Enable strict error checking after environment setup
set -euo pipefail

# ===== 通用优化 =====
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

ulimit -n 32768 || true
ulimit -u 32768 || true

# ===== 根据 CUDA_VISIBLE_DEVICES 计算 GPU 数 =====
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "⚠️  CUDA_VISIBLE_DEVICES 未设置，默认使用 nvidia-smi 检测"
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  # 按逗号切分计算数量
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi
echo "[INFO] Detected ${NUM_GPUS} visible GPUs via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node ${NUM_GPUS} training/training_folsom_mgpu.py