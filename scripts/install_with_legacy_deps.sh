#!/usr/bin/env bash
# Install the project when pip's build isolation breaks for legacy setup.py deps
# (e.g. "setuptools is not available in the build environment" for multitasking/py_vollib).
# Run from repo root: bash scripts/install_with_legacy_deps.sh

set -e
cd "$(dirname "$0")/.."

echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing legacy deps without build isolation (uses current env's setuptools)..."
python -m pip install multitasking --no-build-isolation || true

echo "Installing py_vollib (optional Greeks) without build isolation..."
python -m pip install py_vollib --no-build-isolation || true

echo "Installing project (editable) with optional [dev]..."
python -m pip install -e ".[dev]"

echo "Done. Run: python scripts/run_full_pipeline.py --dry-run"
