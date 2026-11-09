#!/usr/bin/env bash
# One-shot venv setup. Run once after cloning.
# Usage: bash setup.sh

set -euo pipefail

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet      # makes `from veltharrow.X import …` work from anywhere

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run the benchmark:"
echo "  python3 scripts/precompute_retrievals.py --tier small"
echo "  python3 scripts/run_benchmark.py --tier small"
echo "  python3 scripts/run_benchmark.py --tier all      # small + large, sequentially"
echo "  python3 scripts/plot_results.py --tier small"
echo "  pytest -x tests/"
