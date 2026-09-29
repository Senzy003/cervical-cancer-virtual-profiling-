#!/bin/bash
#SBATCH --job-name=vpp-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/features-%j.out
# If the cluster needs a partition name, uncomment and fill in (see `sinfo`):
##SBATCH --partition=gpu

# Extract patch features for every downloaded slide.
#   mkdir -p logs
#   sbatch scripts/extract_features.sh /path/to/slides phikon
# Watch progress:  tail -f logs/features-<jobid>.out
#
# Resumable: finished slides are skipped, so if the job hits its time limit
# (or more slides have downloaded since), just submit it again.

SLIDES="${1:?Usage: sbatch scripts/extract_features.sh <slide folder> [uni|phikon|resnet50]}"
ENCODER="${2:-phikon}"

module load python/3.12.3
source ~/vpp-venv/bin/activate

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

python scripts/extract_features.py \
    --slides "$SLIDES" \
    --cases data/matched_cases.csv \
    --out "features/$ENCODER" \
    --encoder "$ENCODER" \
    --workers 8 \
    --batch-size 128
