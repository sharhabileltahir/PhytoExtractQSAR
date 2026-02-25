#!/usr/bin/env bash
# Master pipeline runner (Bash)
# Usage: bash scripts/run_all.sh

set -euo pipefail

echo "============================================"
echo "Phytochemical Extraction Data Pipeline"
echo "============================================"

echo
echo "[1/7] Verifying compound identities..."
python scripts/01_verify_compounds.py

echo
echo "[2/7] Mining PubMed..."
python scripts/02_mine_pubmed.py

echo
echo "[3/7] Mining PDFs..."
python scripts/03_mine_pdfs.py

echo
echo "[4/7] Replacing synthetic data..."
python scripts/04_replace_synthetic.py

echo
echo "[5/7] Quality control..."
python scripts/05_quality_control.py

echo
echo "[6/7] Computing molecular descriptors..."
python scripts/06_compute_descriptors.py

echo
echo "[7/7] Final export..."
python scripts/07_final_export.py

echo
echo "============================================"
echo "PIPELINE COMPLETE"
echo "Check output/ for final dataset"
echo "Check logs/ for detailed reports"
echo "============================================"
