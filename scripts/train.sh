#!/bin/bash
#SBATCH --job-name=vpp-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%j.out
# If the cluster needs a partition name, uncomment and fill in (see `sinfo`):
##SBATCH --partition=gpu

# Train baseline + attention model with 5-fold cross-validation.
#   sbatch scripts/train.sh phikon            (median split)
#   sbatch scripts/train.sh phikon tertile    (top vs bottom third)
# Results: results/<encoder>[_tertile]/metrics.csv

ENCODER="${1:-phikon}"
SPLIT="${2:-median}"
OUT="results/${ENCODER}"
[ "$SPLIT" != "median" ] && OUT="${OUT}_${SPLIT}"

module load python/3.12.3
source ~/vpp-venv/bin/activate

python scripts/train.py \
    --features "features/$ENCODER" \
    --rppa data/rppa_matrix.csv \
    --out "$OUT" \
    --split "$SPLIT"
