#!/bin/bash
# CPU methods (pseudobulk, composition, gloscope, sampleclr, mixmil) for the full
# benchmark. PaSCient is submitted separately to a GPU partition (see
# ``run_gpu_array.sbatch``).
#
# Submit:
#   sbatch --array=0-9 scripts/aging_benchmark/run_cpu_array.sbatch
#
# Array layout: 2 datasets × 5 CPU methods = 10 cells.

#SBATCH --job-name=aging-cpu
#SBATCH --output=logs/aging_benchmark/full/%A_%a.out
#SBATCH --error=logs/aging_benchmark/full/%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_long
#SBATCH --nice=10000

set -eo pipefail

DATASETS=(aging onek1k)
METHODS=(pseudobulk composition gloscope sampleclr mixmil)

ds_idx=$(( SLURM_ARRAY_TASK_ID / ${#METHODS[@]} ))
m_idx=$(( SLURM_ARRAY_TASK_ID % ${#METHODS[@]} ))
DATASET=${DATASETS[$ds_idx]}
METHOD=${METHODS[$m_idx]}

REPO=/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial
mkdir -p "$REPO/logs/aging_benchmark/full"
cd "$REPO"

echo "[$(date)] cell $SLURM_ARRAY_TASK_ID -> $DATASET / $METHOD on $(hostname)"
echo "  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}"

~/software/miniconda3/envs/patpy/bin/python \
    scripts/aging_benchmark/run_method.py \
    --dataset "$DATASET" \
    --method "$METHOD"

echo "[$(date)] done"
