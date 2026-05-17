#!/bin/bash
# Per-cell attention SampleCLR runs (SSL + FT snapshots) for both cohorts,
# followed by analyse_attention.py for the biology readout.
#
# Submit:
#   sbatch --array=0-1 scripts/aging_benchmark/run_attention_array.sbatch
#
# Array index 0 -> aging, 1 -> onek1k.

#SBATCH --job-name=aging-att
#SBATCH --output=logs/aging_benchmark/full/att_%A_%a.out
#SBATCH --error=logs/aging_benchmark/full/att_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

set -eo pipefail
export PYTHONUNBUFFERED=1

DATASETS=(aging onek1k)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

REPO=/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial
mkdir -p "$REPO/logs/aging_benchmark/full"
cd "$REPO"

echo "[$(date)] cell $SLURM_ARRAY_TASK_ID -> $DATASET on $(hostname)"

PY=~/software/miniconda3/envs/patpy/bin/python
$PY scripts/aging_benchmark/run_sampleclr_attention.py --dataset "$DATASET"
$PY scripts/aging_benchmark/analyze_attention.py        --dataset "$DATASET"

echo "[$(date)] done"
