# Makefile for LibOrtho
# Simple, standard, effective.

PYTHON = python3

.PHONY: help install test benchmark clean

help:
	@echo "LibOrtho Reference Implementation"
	@echo "Targets:"
	@echo "  install   - Install dependencies (from requirements.txt)"
	@echo "  benchmark - Run the core verification script"
	@echo "  clean     - Remove cached files"

install:
	$(PYTHON) -m pip install -r requirements.txt

benchmark:
	$(PYTHON) examples/benchmark.py

clean:
	rm -rf __pycache__
	rm -rf libortho/__pycache__
	rm -f *.pt

