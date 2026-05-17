#!/bin/bash
# GPU runs for PaSCient. The patched PaSCient picks ``cuda`` automatically when
# torch.cuda.is_available(). The CPU full-data extrapolation from the smoke
# was ~5 h/dataset; on GPU we expect 30 min – 2 h.
#
# Submit:
#   sbatch --array=0-1 scripts/aging_benchmark/run_gpu_array.sbatch
#
# Array layout: 2 datasets × 1 method (pascient) = 2 cells.

#SBATCH --job-name=aging-gpu
#SBATCH --output=logs/aging_benchmark/full/gpu_%A_%a.out
#SBATCH --error=logs/aging_benchmark/full/gpu_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

set -eo pipefail

DATASETS=(aging onek1k)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

REPO=/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial
mkdir -p "$REPO/logs/aging_benchmark/full"
cd "$REPO"

echo "[$(date)] cell $SLURM_ARRAY_TASK_ID -> $DATASET / pascient on $(hostname)"
nvidia-smi -L 2>&1 || true

~/software/miniconda3/envs/patpy/bin/python \
    scripts/aging_benchmark/run_method.py \
    --dataset "$DATASET" \
    --method pascient

echo "[$(date)] done"
