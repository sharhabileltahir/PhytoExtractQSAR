# Master pipeline runner (PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "Phytochemical Extraction Data Pipeline"
Write-Host "============================================"

$steps = @(
    @{Label="[1/7] Verifying compound identities..."; Cmd="python scripts/01_verify_compounds.py"},
    @{Label="[2/7] Mining PubMed..."; Cmd="python scripts/02_mine_pubmed.py"},
    @{Label="[3/7] Mining PDFs..."; Cmd="python scripts/03_mine_pdfs.py"},
    @{Label="[4/7] Replacing synthetic data..."; Cmd="python scripts/04_replace_synthetic.py"},
    @{Label="[5/7] Quality control..."; Cmd="python scripts/05_quality_control.py"},
    @{Label="[6/7] Computing molecular descriptors..."; Cmd="python scripts/06_compute_descriptors.py"},
    @{Label="[7/7] Final export..."; Cmd="python scripts/07_final_export.py"}
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host $step.Label
    Invoke-Expression $step.Cmd
}

Write-Host ""
Write-Host "============================================"
Write-Host "PIPELINE COMPLETE"
Write-Host "Check output/ for final dataset"
Write-Host "Check logs/ for detailed reports"
Write-Host "============================================"
