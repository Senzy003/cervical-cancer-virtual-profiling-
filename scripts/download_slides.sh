#!/bin/bash
# Download the matched TCGA-CESC diagnostic slides with gdc-client.
#
# Run inside tmux on the LOGIN node (downloads need internet, not a GPU):
#   tmux new -s download
#   bash scripts/download_slides.sh /path/to/large/storage/slides
#   (detach with Ctrl+B then D; reattach with: tmux attach -t download)
#
# Safe to re-run: gdc-client skips slides that finished downloading.

set -euo pipefail

SLIDE_DIR="${1:?Usage: bash scripts/download_slides.sh <slide folder on large storage>}"
MANIFEST="data/manifests/slides_manifest.txt"
GDC_CLIENT="${GDC_CLIENT:-tools/gdc-client}"

if [ ! -f "$MANIFEST" ]; then
    echo "No manifest yet. Run first: python scripts/prepare_data.py --out data"
    exit 1
fi

if [ ! -x "$GDC_CLIENT" ]; then
    echo "gdc-client not found at $GDC_CLIENT."
    echo "Download the Ubuntu build from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool,"
    echo "unzip it into tools/, then: chmod +x tools/gdc-client"
    exit 1
fi

mkdir -p "$SLIDE_DIR"
echo "Free space where slides will go:"
df -h "$SLIDE_DIR"
echo "Slides to download: $(($(wc -l < "$MANIFEST") - 1))"

"$GDC_CLIENT" download \
    -m "$MANIFEST" \
    -d "$SLIDE_DIR" \
    -n 8 \
    --retry-amount 5 \
    --log-file "$SLIDE_DIR/download.log"

echo "Done. Slides downloaded: $(find "$SLIDE_DIR" -name '*.svs' | wc -l)"
