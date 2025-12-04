# Makefile for LibOrtho
# Simple, standard, effective.

PYTHON = python3

.PHONY: help install test benchmark test-llama clean

help:
	@echo "LibOrtho Reference Implementation"
	@echo "Targets:"
	@echo "  install     - Install dependencies (from requirements.txt)"
	@echo "  benchmark   - Run the core verification script"
	@echo "  test-llama  - Run Llama-3.2-3B test (requires model at /dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B)"
	@echo "  clean       - Remove cached files"

install:
	$(PYTHON) -m pip install -r requirements.txt

benchmark:
	$(PYTHON) examples/benchmark.py

test-llama:
	$(PYTHON) examples/test_llama.py

clean:
	rm -rf __pycache__
	rm -rf libortho/__pycache__
	rm -f *.pt

