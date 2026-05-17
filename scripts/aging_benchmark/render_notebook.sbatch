#!/bin/bash
# Render the standalone age_prediction.ipynb end-to-end on a GPU node.
# Pseudobulk / composition / GloScope / SampleCLR run fine on CPU; PaSCient
# is the one that really benefits from cuda.
#
# Submit:
#   sbatch scripts/aging_benchmark/render_notebook.sbatch

#SBATCH --job-name=aging-render
#SBATCH --output=logs/aging_benchmark/render.%j.out
#SBATCH --error=logs/aging_benchmark/render.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO=/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial
mkdir -p "$REPO/logs/aging_benchmark"
cd "$REPO"

NB="docs/tutorials/notebooks/age_prediction.ipynb"

echo "[$(date)] rendering $NB on $(hostname)"
nvidia-smi -L 2>&1 || true

~/software/miniconda3/envs/patpy/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=14400 \
    "$NB"

echo "[$(date)] done"
