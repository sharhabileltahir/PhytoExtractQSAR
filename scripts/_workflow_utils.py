#!/usr/bin/env python3
"""Shared helpers for the phytochemical extraction workflow scripts."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Phytochemical_Extraction_10K_Dataset.xlsx"
EXTRACTION_SHEET_NAME = "Extraction_Data"

Candidate = Union[str, Tuple[str, ...]]


def normalize_label(value: object) -> str:
    """Normalize a column label for robust matching across encoding variations."""
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace("\u00b0", " degree ").replace("\u00ba", " degree ")
    text = text.replace("\u00b5", " micro ").replace("\u03bc", " micro ")

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_column(columns: Iterable[object], candidates: Sequence[Candidate]) -> Optional[str]:
    """Find a column by exact alias match first, then token containment."""
    normalized = {str(col): normalize_label(col) for col in columns}

    for candidate in candidates:
        if isinstance(candidate, str):
            target = normalize_label(candidate)
            for col_name, col_norm in normalized.items():
                if col_norm == target:
                    return col_name

    for candidate in candidates:
        if isinstance(candidate, tuple):
            tokens = [normalize_label(token) for token in candidate if normalize_label(token)]
            if not tokens:
                continue
            for col_name, col_norm in normalized.items():
                if all(token in col_norm for token in tokens):
                    return col_name

    return None


def ensure_parent_dir(path: Union[str, Path]) -> None:
    """Create a parent directory for a target file if needed."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Union[str, Path]) -> Dict[str, str]:
    """Load key/value pairs from a .env file without extra dependencies."""
    result: Dict[str, str] = {}
    path = Path(env_path)

    if not path.exists():
        return result

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value

    return result


EXTRACTION_COLUMN_CANDIDATES: Dict[str, List[Candidate]] = {
    "id": ["ID", ("id",)],
    "doi": ["DOI / Reference", ("doi", "reference")],
    "year": ["Year", ("year",)],
    "name": ["Phytochemical Name", ("phytochemical", "name")],
    "class": ["Phytochemical Class", ("phytochemical", "class")],
    "smiles": ["SMILES", ("smiles",)],
    "cas": ["CAS Number", ("cas", "number")],
    "plant_source": ["Plant Source (Latin)", ("plant", "source", "latin")],
    "plant_part": ["Plant Part", ("plant", "part")],
    "pretreatment": ["Plant Pretreatment", ("plant", "pretreatment")],
    "method": ["Extraction Method", ("extraction", "method")],
    "solvent": ["Solvent System", ("solvent", "system")],
    "solvent_ratio": ["Solvent Ratio (if mixed)", ("solvent", "ratio")],
    "solvent_volume": ["Solvent Volume (mL/g plant)", ("solvent", "volume")],
    "temperature": ["Temperature (degreeC)", "Temperature (C)", ("temperature",)],
    "time": ["Time (min)", ("time", "min")],
    "pressure": ["Pressure (MPa)", ("pressure", "mpa")],
    "power": ["Power (W)", ("power",)],
    "frequency": ["Frequency (kHz)", ("frequency", "khz")],
    "solid_liquid_ratio": ["Solid:Liquid Ratio", ("solid", "liquid", "ratio")],
    "ph": ["pH", ("ph",)],
    "cycles": ["Number of Cycles", ("number", "cycles")],
    "yield": ["Yield (%)", ("yield",)],
    "purity": ["Purity (%)", ("purity",)],
    "tpc": ["TPC (mg GAE/g)", ("tpc",), ("total", "phenolic")],
    "tfc": ["TFC (mg QE/g)", ("tfc",), ("total", "flavonoid")],
    "ic50": [
        "Antioxidant Activity (IC50, microg/mL)",
        ("antioxidant", "ic50"),
        ("ic50",),
    ],
    "scale": ["Scale (Lab/Pilot/Industrial)", ("scale",)],
    "notes": ["Notes", ("notes",)],
}


def resolve_extraction_columns(columns: Iterable[object]) -> Dict[str, Optional[str]]:
    """Resolve workbook extraction columns to stable internal field names."""
    resolved: Dict[str, Optional[str]] = {}
    for key, candidates in EXTRACTION_COLUMN_CANDIDATES.items():
        resolved[key] = find_column(columns, candidates)
    return resolved


def load_extraction_dataset(path: Union[str, Path] = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the extraction sheet with the known header row offset."""
    return pd.read_excel(path, sheet_name=EXTRACTION_SHEET_NAME, header=2)


def normalize_nullable_text(value: object) -> str:
    """Convert nullable values to a stripped string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_number(value: object) -> Optional[float]:
    """Best-effort numeric coercion for heterogeneous mined values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # Keep only the first numeric token when text includes units.
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def path_or_default(candidates: Sequence[Path], default: Path) -> Path:
    """Return the first existing path from candidates, else default."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default
