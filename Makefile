.PHONY: help setup test precompute bench plot status lint clean

TIER ?= medium

help:
	@echo "Targets:"
	@echo "  make setup                       # venv + deps + editable install"
	@echo "  make test                        # pytest"
	@echo "  make precompute TIER=medium      # phase 1"
	@echo "  make bench TIER=medium           # MCQ benchmark"
	@echo "  make plot TIER=medium            # generate all plots"
	@echo "  make status                      # refresh README status table from results/"
	@echo "  make clean                       # nuke cache + matplotlib config"

setup:
	bash setup.sh

test:
	pytest -x tests/

precompute:
	python3 scripts/precompute_retrievals.py --tier $(TIER)

bench:
	python3 scripts/run_benchmark.py --tier $(TIER)

plot:
	python3 scripts/plot_results.py --tier $(TIER)

status:
	python3 tools/status_board.py

clean:
	rm -rf cache/.matplotlib cache/embeddings/*.npz cache/chunks/*.jsonl
