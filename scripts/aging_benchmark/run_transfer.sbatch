#!/bin/bash
# Cross-cohort transfer: train PaSCient + SampleCLR on AIFI in a shared PCA
# space, predict OneK1K ages with the resulting models.
#
# Submit:
#   sbatch scripts/aging_benchmark/run_transfer.sbatch

#SBATCH --job-name=aging-transfer
#SBATCH --output=logs/aging_benchmark/transfer.%j.out
#SBATCH --error=logs/aging_benchmark/transfer.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO=/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial
mkdir -p "$REPO/logs/aging_benchmark"
cd "$REPO"

echo "[$(date)] cross-cohort transfer on $(hostname)"
nvidia-smi -L 2>&1 || true

~/software/miniconda3/envs/patpy/bin/python \
    scripts/aging_benchmark/cross_cohort_transfer.py

echo "[$(date)] done"
