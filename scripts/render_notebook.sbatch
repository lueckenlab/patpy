#!/bin/bash
#SBATCH --job-name=render_supervised_nb
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/render_notebook_%j.log
#SBATCH --error=logs/render_notebook_%j.log
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

set -euo pipefail

REPO_DIR="/lustre/groups/luckylab/workspace/vladimir.shitov/patpy"
NOTEBOOK="docs/notebooks/pascient_example.ipynb"

cd "$REPO_DIR"
mkdir -p logs

echo "=== $(date) === Pulling latest code ==="
conda run -n patpy pip install -e . --quiet
echo "=== $(date) === Starting notebook render ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run -n patpy jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output pascient_example.ipynb \
    "$NOTEBOOK"

echo "=== $(date) === Notebook render complete ==="
