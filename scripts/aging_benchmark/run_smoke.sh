#!/bin/bash
# Run all 12 (dataset x method) cells on a 20-donor x 200-cell smoke subset.
#
# Sequentially because a smoke run is supposed to be fast; ~2-5 min per cell
# on CPU. Logs land under logs/aging_benchmark/smoke/.

set -u
cd "$(dirname "$0")/../.."   # to repo root

LOG_DIR=logs/aging_benchmark/smoke
mkdir -p "$LOG_DIR"

PY=~/software/miniconda3/envs/patpy/bin/python

DATASETS=(aging onek1k)
METHODS=(pseudobulk composition gloscope pascient mixmil sampleclr)

for ds in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        echo "=== $ds / $m ==="
        log="$LOG_DIR/${ds}_${m}.log"
        $PY scripts/aging_benchmark/run_method.py --dataset "$ds" --method "$m" --smoke \
            >"$log" 2>&1
        rc=$?
        tail -n 3 "$log"
        echo "  rc=$rc  log=$log"
    done
done
echo "DONE"
