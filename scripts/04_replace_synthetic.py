#!/usr/bin/env python3
"""Phase 4: replace synthetic dataset rows with mined literature values."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts._workflow_utils import (
    PROJECT_ROOT,
    ensure_parent_dir,
    find_column,
    load_extraction_dataset,
    normalize_label,
    normalize_nullable_text,
    path_or_default,
    resolve_extraction_columns,
    to_number,
)

LOG_PATH = PROJECT_ROOT / "logs" / "04_replacement.log"
REPORT_PATH = PROJECT_ROOT / "logs" / "replacement_report.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_with_replacements.xlsx"

BASE_DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "dataset_with_replacements.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Phytochemical_Extraction_10K_Dataset.xlsx",
]

MINED_FILES = [
    PROJECT_ROOT / "data" / "processed" / "mined_pubmed_data.xlsx",
    PROJECT_ROOT / "data" / "processed" / "mined_pdf_data.xlsx",
    PROJECT_ROOT / "data" / "verified" / "verified_entries.xlsx",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CONFIDENCE_ORDER = {"verified": 0, "high": 1, "medium": 2, "low": 3, "synthetic": 9}


def _load_base_dataset() -> Tuple[pd.DataFrame, Path]:
    base_path = path_or_default(
        [BASE_DATASET_CANDIDATES[0]],
        BASE_DATASET_CANDIDATES[1],
    )

    if base_path == BASE_DATASET_CANDIDATES[1]:
        df = load_extraction_dataset(base_path)
    else:
        df = pd.read_excel(base_path)

    return df, base_path


def _standardize_mined_columns(frame: pd.DataFrame) -> pd.DataFrame:
    candidates: Dict[str, List[object]] = {
        "compound": ["compound", "Phytochemical Name", ("phytochemical", "name")],
        "compound_hint": ["compound_hint", ("compound", "hint")],
        "doi": ["doi", "DOI / Reference", ("doi", "reference")],
        "pmid": ["pmid", ("pmid",)],
        "year": ["year", ("year",)],
        "title": ["title", ("title",)],
        "method": ["method", "Extraction Method", ("extraction", "method")],
        "solvent": ["solvent", "Solvent System", ("solvent", "system")],
        "temperature_c": ["temperature_c", "Temperature (C)", ("temperature",)],
        "time_min": ["time_min", "Time (min)", ("time", "min")],
        "pressure_mpa": ["pressure_mpa", "Pressure (MPa)", ("pressure", "mpa")],
        "power_w": ["power_w", "Power (W)", ("power",)],
        "frequency_khz": ["frequency_khz", "Frequency (kHz)", ("frequency", "khz")],
        "yield_pct": ["yield_pct", "Yield (%)", ("yield",)],
        "tpc": ["tpc", "TPC (mg GAE/g)", ("tpc",), ("total", "phenolic")],
        "tfc": ["tfc", "TFC (mg QE/g)", ("tfc",), ("total", "flavonoid")],
        "ic50": ["ic50", ("antioxidant", "ic50"), ("ic50",)],
        "solid_liquid_ratio": [
            "solid_liquid_ratio",
            "Solid:Liquid Ratio",
            ("solid", "liquid", "ratio"),
        ],
        "source": ["source", ("source",)],
        "confidence": ["confidence", ("confidence",)],
        "source_file": ["source_file", ("source", "file")],
    }

    out = pd.DataFrame(index=frame.index)
    for target_name, target_candidates in candidates.items():
        src_col = find_column(frame.columns, target_candidates)
        if src_col:
            out[target_name] = frame[src_col]
        else:
            out[target_name] = pd.NA

    out["compound"] = out["compound"].fillna("").map(normalize_nullable_text)
    out["compound_hint"] = out["compound_hint"].fillna("").map(normalize_nullable_text)

    missing_compound = out["compound"].eq("") & out["compound_hint"].ne("")
    out.loc[missing_compound, "compound"] = out.loc[missing_compound, "compound_hint"]

    out["doi"] = out["doi"].fillna("").map(normalize_nullable_text)
    out["pmid"] = out["pmid"].fillna("").map(normalize_nullable_text)
    out["source"] = out["source"].fillna("").map(normalize_nullable_text).replace("", "mined")
    out["confidence"] = (
        out["confidence"].fillna("").map(normalize_nullable_text).str.lower().replace("", "medium")
    )

    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "temperature_c",
        "time_min",
        "pressure_mpa",
        "power_w",
        "frequency_khz",
        "yield_pct",
        "tpc",
        "tfc",
        "ic50",
    ]
    for column in numeric_cols:
        out[column] = out[column].apply(to_number)

    out["solid_liquid_ratio"] = out["solid_liquid_ratio"].fillna("").map(normalize_nullable_text)
    out["_compound_norm"] = out["compound"].map(normalize_label)
    out["_confidence_rank"] = out["confidence"].map(lambda x: CONFIDENCE_ORDER.get(x, 7))

    # Score rows by how much useful data they carry.
    richness_fields = [
        "doi",
        "year",
        "method",
        "solvent",
        "temperature_c",
        "time_min",
        "yield_pct",
        "tpc",
        "tfc",
        "ic50",
    ]
    out["_richness"] = (
        out[richness_fields]
        .notna()
        .astype(int)
        .sum(axis=1)
        + out["doi"].ne("").astype(int)
    )

    return out


def load_mined_data() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in MINED_FILES:
        if not path.exists():
            continue

        frame = pd.read_excel(path)
        standardized = _standardize_mined_columns(frame)

        if path.name == "verified_entries.xlsx":
            standardized["confidence"] = "verified"
            standardized["_confidence_rank"] = CONFIDENCE_ORDER["verified"]
            standardized["source"] = standardized["source"].replace("mined", "manual_verified")

        frames.append(standardized)
        logger.info("Loaded %s records from %s", len(standardized), path)

    if not frames:
        return pd.DataFrame()

    mined = pd.concat(frames, ignore_index=True)
    mined = mined[mined["_compound_norm"].ne("")].copy()
    mined = mined.sort_values(["_confidence_rank", "_richness"], ascending=[True, False])
    mined = mined.reset_index(drop=True)
    return mined


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def replace_entries(df_base: pd.DataFrame, mined: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df_base.copy()
    cols = resolve_extraction_columns(df.columns)

    if not cols["name"]:
        raise RuntimeError("Could not resolve 'Phytochemical Name' column in base dataset")

    if "_data_source" not in df.columns:
        df["_data_source"] = "synthetic"
    if "_confidence" not in df.columns:
        df["_confidence"] = "synthetic"
    if "_replacement_date" not in df.columns:
        df["_replacement_date"] = pd.NaT
    if "_original_doi" not in df.columns:
        doi_col = cols["doi"]
        if doi_col:
            df["_original_doi"] = df[doi_col]
        else:
            df["_original_doi"] = ""

    name_col = cols["name"]
    df["_compound_norm"] = df[name_col].fillna("").map(normalize_label)

    replacement_count = 0

    field_map = {
        "method": cols["method"],
        "solvent": cols["solvent"],
        "temperature_c": cols["temperature"],
        "time_min": cols["time"],
        "pressure_mpa": cols["pressure"],
        "power_w": cols["power"],
        "frequency_khz": cols["frequency"],
        "yield_pct": cols["yield"],
        "tpc": cols["tpc"],
        "tfc": cols["tfc"],
        "ic50": cols["ic50"],
        "solid_liquid_ratio": cols["solid_liquid_ratio"],
    }

    numeric_fields = {
        "temperature_c",
        "time_min",
        "pressure_mpa",
        "power_w",
        "frequency_khz",
        "yield_pct",
        "tpc",
        "tfc",
        "ic50",
    }

    # Some text columns can be inferred as float when mostly empty; coerce to object
    # before writing replacement strings.
    text_target_cols = [
        base_col
        for mined_field, base_col in field_map.items()
        if base_col and mined_field not in numeric_fields
    ]
    doi_col = cols["doi"]
    notes_col = cols["notes"]
    for col_name in [*text_target_cols, doi_col, notes_col]:
        if col_name and col_name in df.columns and df[col_name].dtype != "object":
            df[col_name] = df[col_name].astype("object")

    for _, mined_row in mined.iterrows():
        compound_norm = mined_row.get("_compound_norm", "")
        if not compound_norm:
            continue

        mask = (df["_compound_norm"] == compound_norm) & (df["_data_source"] == "synthetic")
        matches = df.index[mask]
        if len(matches) == 0:
            continue

        row_idx = matches[0]

        if doi_col and not _is_blank(mined_row.get("doi")):
            df.at[row_idx, doi_col] = normalize_nullable_text(mined_row.get("doi"))

        year_col = cols["year"]
        mined_year = int(mined_row.get("year", 0)) if not pd.isna(mined_row.get("year")) else 0
        if year_col and mined_year > 0:
            df.at[row_idx, year_col] = mined_year

        for mined_field, base_col in field_map.items():
            if not base_col:
                continue

            mined_value = mined_row.get(mined_field)
            if _is_blank(mined_value):
                continue

            if mined_field in numeric_fields:
                numeric_value = to_number(mined_value)
                if numeric_value is None:
                    continue
                df.at[row_idx, base_col] = numeric_value
            else:
                df.at[row_idx, base_col] = normalize_nullable_text(mined_value)

        df.at[row_idx, "_data_source"] = normalize_nullable_text(mined_row.get("source", "mined"))
        df.at[row_idx, "_confidence"] = normalize_nullable_text(mined_row.get("confidence", "medium"))
        df.at[row_idx, "_replacement_date"] = pd.Timestamp.now()

        pmid = normalize_nullable_text(mined_row.get("pmid", ""))
        source_file = normalize_nullable_text(mined_row.get("source_file", ""))
        if notes_col and (pmid or source_file):
            provenance_parts = []
            if pmid:
                provenance_parts.append(f"PMID:{pmid}")
            if source_file:
                provenance_parts.append(f"PDF:{source_file}")
            provenance = f"Auto replacement ({' | '.join(provenance_parts)})"
            existing_note = normalize_nullable_text(df.at[row_idx, notes_col])
            df.at[row_idx, notes_col] = f"{existing_note}; {provenance}" if existing_note else provenance

        replacement_count += 1

    df = df.drop(columns=["_compound_norm"], errors="ignore")
    return df, replacement_count


def main() -> None:
    base_df, base_path = _load_base_dataset()
    logger.info("Loaded base dataset: %s (%s rows)", base_path, len(base_df))

    mined = load_mined_data()
    if mined.empty:
        logger.warning("No mined/verified data found. Run phases 2-3 or add verified entries first.")
        return

    logger.info("Total mined records available for replacement: %s", len(mined))

    replaced_df, replacement_count = replace_entries(base_df, mined)

    ensure_parent_dir(OUTPUT_PATH)
    replaced_df.to_excel(OUTPUT_PATH, index=False)

    total_rows = len(replaced_df)
    replaced_rows = int((replaced_df["_data_source"] != "synthetic").sum())
    verified_rows = int((replaced_df["_confidence"].str.lower() == "verified").sum())
    high_rows = int((replaced_df["_confidence"].str.lower() == "high").sum())
    medium_rows = int((replaced_df["_confidence"].str.lower() == "medium").sum())

    report = (
        "=" * 60
        + "\nREPLACEMENT REPORT\n"
        + "=" * 60
        + f"\nBase dataset:            {base_path}"
        + f"\nTotal rows:              {total_rows}"
        + f"\nRows replaced this run:  {replacement_count}"
        + f"\nReplaced total:          {replaced_rows} ({(100 * replaced_rows / total_rows):.1f}%)"
        + f"\n  - Manually verified:   {verified_rows}"
        + f"\n  - High confidence:     {high_rows}"
        + f"\n  - Medium confidence:   {medium_rows}"
        + f"\nStill synthetic:         {total_rows - replaced_rows} ({(100 * (total_rows - replaced_rows) / total_rows):.1f}%)"
        + f"\n\nOutput: {OUTPUT_PATH}"
        + "\n" + "=" * 60 + "\n"
    )

    logger.info("\n%s", report)
    ensure_parent_dir(REPORT_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
