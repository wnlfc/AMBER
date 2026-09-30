#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RETRIEVAL_JSON:?Set RETRIEVAL_JSON to one PhotoBench subset/language retrieval JSON.}"
: "${IMAGE_DIR:?Set IMAGE_DIR to the matching PhotoBench image directory.}"
python3 elo_rerank.py \
  --input "$RETRIEVAL_JSON" \
  --output "${OUTPUT:-outputs/photobench/amber.jsonl}" \
  --image_dir "$IMAGE_DIR" \
  --api_url "${API_URL:-http://localhost:8005/v1}" \
  --model "${MODEL:-qwen3-vl-8b-instruct}" \
  --image_max_size 1120 \
  "$@"

