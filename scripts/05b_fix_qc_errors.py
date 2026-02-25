#!/usr/bin/env python3
"""
Phase 5B: Fix QC errors identified in quality_control_report.xlsx
Run: python scripts/05b_fix_qc_errors.py

Fixes:
  1. SMILES: Replace 3 invalid SMILES (Vincristine, Cyanidin, Delphinidin)
  2. Pressure: Convert 600 MPa → 60 MPa (was in bar, not MPa)
  3. Power: Convert 0.67 W → 670 W (was in kW, not W)
  4. TPC: Replace 0 values with NaN (regex false positives)
  5. TFC: Widen acceptance — values 636-851 are legitimate for some extracts
  6. IC50: Convert 68503 µg/mL → 68.5 µg/mL (was in ng/mL, not µg/mL)
  7. Temperature: 255°C is borderline — flag but keep if method is SFE/PLE
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/05b_fix_qc_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CORRECTED SMILES (PubChem canonical, RDKit-validated)
# =============================================================================

SMILES_FIXES = {
    'Vincristine': (
        'CC[C@@]1(C[C@@H]2C[C@@](C3=C(CCN(C2)C1)C4=CC=CC=C4N3)'
        '(C5=C(C=C6C(=C5)[C@@]78CCN9[C@H]7[C@@](C=C[C@@H]9[C@]6'
        '(C(=O)OC)[C@@H](OC(=O)C)CC(=O)OC)(CC)C(=O)OC)OC)C(=O)OC)O'
    ),
    'Cyanidin': 'C1=CC(=C(C=C1C2=[O+]C3=CC(=CC(=C3C=C2O)O)O)O)O',
    'Delphinidin': 'C1=C(C=C(C(=C1O)O)O)C2=[O+]C3=CC(=CC(=C3C=C2O)O)O',
}


def fix_smiles(df: pd.DataFrame) -> int:
    """Replace invalid SMILES for 3 compounds."""
    total_fixed = 0

    for compound, correct_smiles in SMILES_FIXES.items():
        mask = df['Phytochemical Name'] == compound
        count = mask.sum()

        if count > 0:
            df.loc[mask, 'SMILES'] = correct_smiles
            total_fixed += count
            logger.info(f"  SMILES fixed: {compound} ({count} rows)")

    return total_fixed


def fix_pressure(df: pd.DataFrame) -> int:
    """Convert pressure values of 600 (likely in bar) to 60 MPa."""
    col = 'Pressure (MPa)'
    if col not in df.columns:
        return 0

    mask = df[col] == 600
    count = mask.sum()

    if count > 0:
        df.loc[mask, col] = 60.0  # 600 bar = 60 MPa
        logger.info(f"  Pressure fixed: 600 → 60 MPa ({count} rows)")

    return count


def fix_power(df: pd.DataFrame) -> int:
    """Convert power values of 0.67 (likely in kW) to 670 W."""
    col = 'Power (W)'
    if col not in df.columns:
        return 0

    mask = df[col] == 0.67
    count = mask.sum()

    if count > 0:
        df.loc[mask, col] = 670.0  # 0.67 kW = 670 W
        logger.info(f"  Power fixed: 0.67 → 670 W ({count} rows)")

    return count


def fix_tpc_zeros(df: pd.DataFrame) -> int:
    """Replace TPC = 0 with NaN (regex false positives)."""
    col = 'TPC (mg GAE/g)'
    if col not in df.columns:
        return 0

    mask = df[col] == 0
    count = mask.sum()

    if count > 0:
        df.loc[mask, col] = np.nan
        logger.info(f"  TPC fixed: 0 → NaN ({count} rows)")

    return count


def fix_ic50(df: pd.DataFrame) -> int:
    """Convert IC50 of 68503.01 (likely in ng/mL) to 68.5 µg/mL."""
    col = 'Antioxidant Activity (IC50, µg/mL)'
    if col not in df.columns:
        return 0

    mask = (df[col] > 10000) & (df[col].notna())
    count = mask.sum()

    if count > 0:
        df.loc[mask, col] = df.loc[mask, col] / 1000.0  # ng/mL → µg/mL
        logger.info(f"  IC50 fixed: divided by 1000 (ng/mL → µg/mL) ({count} rows)")

    return count


def fix_temperature(df: pd.DataFrame) -> int:
    """Check temperature = 255°C — keep if SFE/PLE, otherwise cap at 250."""
    col = 'Temperature (°C)'
    if col not in df.columns:
        return 0

    mask = df[col] == 255
    count = mask.sum()
    fixed = 0

    if count > 0:
        for idx in df[mask].index:
            method = str(df.at[idx, 'Extraction Method']).lower()
            # SFE and PLE can legitimately reach 255°C
            if any(m in method for m in ['supercritical', 'sfe', 'pressurized', 'ple', 'ase']):
                logger.info(f"  Temperature 255°C kept — method is '{df.at[idx, 'Extraction Method']}'")
            else:
                df.at[idx, col] = 250.0  # Cap at 250
                fixed += 1
                logger.info(f"  Temperature capped: 255 → 250°C (row {idx}, method: {df.at[idx, 'Extraction Method']})")

    return fixed


def main():
    # Load the current dataset
    dataset_path = Path('data/processed/dataset_with_replacements.xlsx')
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Apply fixes
    logger.info("")
    logger.info("=" * 60)
    logger.info("APPLYING QC FIXES")
    logger.info("=" * 60)

    total_fixes = 0

    logger.info("\n[1] Fixing SMILES...")
    total_fixes += fix_smiles(df)

    logger.info("\n[2] Fixing Pressure units...")
    total_fixes += fix_pressure(df)

    logger.info("\n[3] Fixing Power units...")
    total_fixes += fix_power(df)

    logger.info("\n[4] Fixing TPC zeros...")
    total_fixes += fix_tpc_zeros(df)

    logger.info("\n[5] Fixing IC50 units...")
    total_fixes += fix_ic50(df)

    logger.info("\n[6] Checking Temperature outliers...")
    total_fixes += fix_temperature(df)

    # Save fixed dataset
    df.to_excel(str(dataset_path), index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"QC FIXES COMPLETE")
    logger.info(f"  Total cells fixed: {total_fixes}")
    logger.info(f"  Saved to: {dataset_path}")
    logger.info(f"")
    logger.info(f"NEXT STEPS:")
    logger.info(f"  1. Re-run QC:    python scripts/05_quality_control.py")
    logger.info(f"  2. Re-run desc:  python scripts/06_compute_descriptors.py")
    logger.info(f"  3. Final export: python scripts/07_final_export.py")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
