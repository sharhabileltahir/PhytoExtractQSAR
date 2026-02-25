# PhytoExtractQSAR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

An automated cheminformatics pipeline for literature-mined QSAR modeling of phytochemical extraction outcomes.

## Overview

PhytoExtractQSAR is an end-to-end pipeline that automates:
1. **Literature Mining** - Mining phytochemical extraction data from PubMed abstracts and full-text PDFs
2. **Molecular Descriptors** - Computing molecular descriptors from validated SMILES using RDKit
3. **Feature Engineering** - Engineering physicochemically meaningful features including solvent polarity descriptors and compound–method interaction terms
4. **QSAR Modeling** - Training and validating multi-target QSAR models with transparent data provenance tracking

## Key Features

- **Automated data collection** from PubMed (3,322 records) and PDF extraction (810 PDFs)
- **1,877 literature-verified experimental records** across 94 phytochemical compounds and 15 extraction methods
- **Multi-target QSAR models** for yield, TPC, TFC, and IC50 prediction
- **Transparent data provenance** tracking (literature-verified vs. computationally augmented)
- **Applicability domain assessment** using leverage-based Williams plots
- **SHAP-based feature importance** analysis for model interpretability

## Model Performance

| Target | Best Model | N | Q² | Pearson r |
|--------|-----------|---|----|-----------|
| TFC (mg QE/g) | Extra Trees | 1,022 | 0.583 | 0.692 |
| Yield (%) | Extra Trees | 1,877 | 0.341 | 0.562 |
| TPC (mg GAE/g) | Random Forest | 1,287 | 0.198 | 0.401 |
| IC50 (µg/mL) | Extra Trees | 1,359 | 0.161 | 0.249 |

## Project Structure

```
PhytoExtractQSAR/
├── data/
│   ├── raw/                    # Input datasets
│   ├── processed/              # Intermediate processed data
│   └── pdfs/                   # Downloaded full-text PDFs
├── scripts/
│   ├── 01_verify_compounds.py  # SMILES validation via PubChem
│   ├── 02_mine_pubmed.py       # PubMed abstract mining
│   ├── 03_mine_pdfs.py         # PDF full-text extraction
│   ├── 04_replace_synthetic.py # Data integration
│   ├── 05_quality_control.py   # Range filters and QC
│   ├── 06_compute_descriptors.py # RDKit molecular descriptors
│   ├── 07_final_export.py      # Dataset export
│   ├── 08_targeted_mining.py   # Targeted PubMed queries
│   ├── 09c_qsar_engineered.py  # QSAR modeling with engineered features
│   ├── 10_feature_engineering.py # Domain-specific feature engineering
│   └── generate_manuscript_figures.py # Publication figures
├── output/
│   ├── qsar_results_v3/        # Model results and predictions
│   └── Phytochemical_Extraction_Dataset_FINAL_*.xlsx
├── figures/                    # Publication-ready figures
├── logs/                       # Pipeline execution logs
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PhytoExtractQSAR.git
cd PhytoExtractQSAR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.12+
- RDKit 2023.09+
- scikit-learn 1.5+
- XGBoost 3.2+
- SHAP 0.50+
- pandas, numpy, matplotlib, seaborn

## Usage

### Full Pipeline

```bash
# Run all phases sequentially
python scripts/01_verify_compounds.py
python scripts/02_mine_pubmed.py
python scripts/03_mine_pdfs.py
python scripts/04_replace_synthetic.py
python scripts/05_quality_control.py
python scripts/06_compute_descriptors.py
python scripts/07_final_export.py
python scripts/08_targeted_mining.py
python scripts/10_feature_engineering.py
python scripts/09c_qsar_engineered.py
```

### Generate Figures

```bash
python scripts/generate_manuscript_figures.py
```

### Environment Variables

Create a `.env` file in the project root:

```bash
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_api_key
```

## Data Provenance

The pipeline maintains explicit provenance tracking for every record:
- **PubMed abstract mining** - Data extracted from PubMed abstracts
- **PDF full-text extraction** - Data extracted from downloaded PDFs
- **Computationally augmented** - Records flagged separately (excluded from model training)

All QSAR models are trained exclusively on literature-verified records using nested 5-fold cross-validation.

## Figures

The pipeline generates publication-ready figures:

- **Figure 1**: Dataset composition (compound classes, extraction methods)
- **Figure 2**: Data scaling analysis (Q² vs sample size)
- **Figure 3**: SHAP feature importance beeswarm plots
- **Figure 4**: Williams plots for applicability domain assessment

## Citation

If you use PhytoExtractQSAR in your research, please cite:

```bibtex
@article{PhytoExtractQSAR2026,
  title={An Automated Cheminformatics Pipeline for Literature-Mined QSAR Modeling of Phytochemical Extraction Outcomes},
  author={[Author names]},
  journal={Journal of Cheminformatics},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RDKit: Open-source cheminformatics (https://www.rdkit.org)
- PubChem: Chemical database (https://pubchem.ncbi.nlm.nih.gov)
- NCBI Entrez: Literature access API
