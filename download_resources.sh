#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/data"}"
MODELS_DIR="$ROOT_DIR/models"

mkdir -p "$OUT_DIR" "$MODELS_DIR"

URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectorizer.joblib"
DEST="$MODELS_DIR/$(basename "$URL")"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$DEST" "$URL"
else
  echo "Error: install curl or wget." >&2
  exit 1
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Aremaki/BEL_resources \
    --repo-type dataset \
    --local-dir "$OUT_DIR" \
    --include "termino_raw/**"
else
  echo "huggingface-cli not found. Install with: pip install -U huggingface_hub" >&2
fi

echo "Done: $OUT_DIR"
