#!/usr/bin/env python3
"""Update dataset SMILES using PubChem-verified values while preserving CAS values.

Rules implemented:
1. Replace compound SMILES with `verified_smiles` from Phase 1 report for all verified compounds.
2. Keep original CAS numbers unchanged, including known CAS mismatches.
3. Manually set Farnesol SMILES to PubChem CID 445070 value:
   CC(=CCCC(=CCCC(=CCO)C)C)C
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import (
    PROJECT_ROOT,
    load_extraction_dataset,
    normalize_label,
    normalize_nullable_text,
    path_or_default,
    resolve_extraction_columns,
)

LOG_PATH = PROJECT_ROOT / "logs" / "08_update_smiles.log"
TEXT_REPORT_PATH = PROJECT_ROOT / "logs" / "08_smiles_update_report.txt"
VERIFICATION_PATH = PROJECT_ROOT / "data" / "processed" / "compound_verification_report.xlsx"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_with_standardized_smiles.xlsx"
OUTPUT_AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "smiles_update_audit.xlsx"

MANUAL_FARNESOL_SMILES = "CC(=CCCC(=CCCC(=CCO)C)C)C"
MANUAL_FARNESOL_CID = 445070

DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "dataset_with_replacements.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Phytochemical_Extraction_10K_Dataset.xlsx",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _load_input_dataset() -> Tuple[pd.DataFrame, Path]:
    dataset_path = path_or_default([DATASET_CANDIDATES[0]], DATASET_CANDIDATES[1])
    if dataset_path == DATASET_CANDIDATES[1]:
        return load_extraction_dataset(dataset_path), dataset_path
    return pd.read_excel(dataset_path), dataset_path


def _build_smiles_map(df_verify: pd.DataFrame) -> Tuple[Dict[str, str], int]:
    required_cols = {"original_name", "verified", "verified_smiles"}
    missing = required_cols.difference(df_verify.columns)
    if missing:
        raise RuntimeError(f"Verification report missing columns: {sorted(missing)}")

    verified_rows = df_verify[(df_verify["verified"] == True)].copy()
    verified_count = len(verified_rows)

    smiles_map: Dict[str, str] = {}
    for _, row in verified_rows.iterrows():
        name = normalize_nullable_text(row["original_name"])
        smiles = normalize_nullable_text(row["verified_smiles"])
        if not name or not smiles:
            continue
        smiles_map[normalize_label(name)] = smiles

    # Manual override for the one timeout case from Phase 1.
    smiles_map[normalize_label("Farnesol")] = MANUAL_FARNESOL_SMILES

    return smiles_map, verified_count


def _extract_cas_mismatches(df_verify: pd.DataFrame) -> pd.DataFrame:
    cols_needed = {"original_name", "original_cas", "verified_cas", "verified"}
    if not cols_needed.issubset(df_verify.columns):
        return pd.DataFrame(columns=["original_name", "original_cas", "verified_cas", "action"])

    mask = (df_verify["verified"] == True) & df_verify["verified_cas"].notna() & df_verify["original_cas"].notna()
    subset = df_verify.loc[mask, ["original_name", "original_cas", "verified_cas"]].copy()

    subset["original_cas_norm"] = subset["original_cas"].map(normalize_nullable_text)
    subset["verified_cas_norm"] = subset["verified_cas"].map(normalize_nullable_text)
    subset = subset[subset["original_cas_norm"] != subset["verified_cas_norm"]].copy()
    subset = subset.drop(columns=["original_cas_norm", "verified_cas_norm"])
    subset["action"] = "Kept original CAS from dataset"
    return subset.reset_index(drop=True)


def _series_fingerprint(series: pd.Series) -> pd.Series:
    return series.fillna("<NA>").map(lambda x: normalize_nullable_text(x))


def update_smiles(df: pd.DataFrame, smiles_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = resolve_extraction_columns(df.columns)
    name_col = cols.get("name")
    smiles_col = cols.get("smiles")
    cas_col = cols.get("cas")

    if not name_col or not smiles_col:
        raise RuntimeError("Dataset is missing required columns: Phytochemical Name and/or SMILES")

    df_out = df.copy()
    cas_before = _series_fingerprint(df_out[cas_col]) if cas_col else None

    df_out["_compound_norm"] = df_out[name_col].fillna("").map(normalize_label)
    summary_rows: List[Dict[str, object]] = []

    for norm_name, target_smiles in sorted(smiles_map.items()):
        mask = df_out["_compound_norm"] == norm_name
        matched_rows = int(mask.sum())
        if matched_rows == 0:
            summary_rows.append(
                {
                    "compound_norm": norm_name,
                    "rows_matched": 0,
                    "rows_changed": 0,
                    "status": "not_found_in_dataset",
                }
            )
            continue

        current_smiles = df_out.loc[mask, smiles_col].fillna("").map(normalize_nullable_text)
        changed = current_smiles != target_smiles
        rows_changed = int(changed.sum())

        df_out.loc[mask, smiles_col] = target_smiles

        summary_rows.append(
            {
                "compound_norm": norm_name,
                "rows_matched": matched_rows,
                "rows_changed": rows_changed,
                "status": "updated",
            }
        )

    df_out = df_out.drop(columns=["_compound_norm"], errors="ignore")

    # Safety guard: CAS values must remain untouched.
    if cas_col:
        cas_after = _series_fingerprint(df_out[cas_col])
        if not cas_before.equals(cas_after):
            raise RuntimeError("CAS values changed unexpectedly; aborting to preserve original CAS numbers")

    return df_out, pd.DataFrame(summary_rows)


def main() -> None:
    if not VERIFICATION_PATH.exists():
        raise FileNotFoundError(f"Verification report not found: {VERIFICATION_PATH}")

    df_verify = pd.read_excel(VERIFICATION_PATH)
    df_input, input_path = _load_input_dataset()

    smiles_map, verified_count = _build_smiles_map(df_verify)
    cas_mismatches = _extract_cas_mismatches(df_verify)

    df_updated, df_summary = update_smiles(df_input, smiles_map)

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_excel(OUTPUT_DATASET_PATH, index=False)

    OUTPUT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_AUDIT_PATH, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="smiles_update_summary", index=False)
        cas_mismatches.to_excel(writer, sheet_name="cas_mismatches_kept", index=False)

    updated_compounds = int((df_summary["status"] == "updated").sum()) if not df_summary.empty else 0
    missing_compounds = int((df_summary["status"] == "not_found_in_dataset").sum()) if not df_summary.empty else 0
    rows_changed = int(df_summary["rows_changed"].sum()) if not df_summary.empty else 0
    rows_matched = int(df_summary["rows_matched"].sum()) if not df_summary.empty else 0

    report = (
        "=" * 60
        + "\nSMILES STANDARDIZATION REPORT"
        + "\n" + "=" * 60
        + f"\nInput dataset:              {input_path}"
        + f"\nVerification report:        {VERIFICATION_PATH}"
        + f"\nVerified compounds in report: {verified_count}"
        + f"\nManual Farnesol override:   {MANUAL_FARNESOL_SMILES} (CID {MANUAL_FARNESOL_CID})"
        + f"\nTarget compounds updated:   {len(smiles_map)}"
        + f"\nCompounds found in dataset: {updated_compounds}"
        + f"\nCompounds not found:        {missing_compounds}"
        + f"\nRows matched:               {rows_matched}"
        + f"\nRows with SMILES changed:   {rows_changed}"
        + f"\nCAS mismatches preserved:   {len(cas_mismatches)}"
        + f"\n\nUpdated dataset: {OUTPUT_DATASET_PATH}"
        + f"\nAudit workbook:  {OUTPUT_AUDIT_PATH}"
        + "\n" + "=" * 60
    )

    logger.info("\n%s", report)
    TEXT_REPORT_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
