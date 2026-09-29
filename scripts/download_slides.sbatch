#!/bin/bash
#SBATCH --job-name=vpp-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download-%j.out
bash scripts/download_slides.sh slides/
