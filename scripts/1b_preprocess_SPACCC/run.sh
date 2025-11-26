#!/bin/bash
# This script runs the preprocessing steps for the SPACCC dataset.

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "▶️  Running prepare_terminology.py..."
python "$SCRIPT_DIR/prepare_terminology.py"

echo "▶️  Running prepare_corpus.py..."
python "$SCRIPT_DIR/prepare_corpus.py"

echo "🎉 All SPACCC preprocessing steps completed successfully."
